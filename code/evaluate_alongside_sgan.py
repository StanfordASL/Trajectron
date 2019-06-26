import torch
import numpy as np
import random
import argparse
from utils import plot_utils, eval_utils
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

import pandas as pd


hyperparams = {
    ### Training
    ## Batch Sizes
    'batch_size': 16,
    ## Learning Rate
    'learning_rate': 0.002,
    'min_learning_rate': 0.00001,
    'learning_decay_rate': 0.9999,
    ## Optimizer
    # 'optimizer': tf.train.AdamOptimizer,
    'optimizer_kwargs': {},
    'grad_clip': 1.0,

    ### Prediction
    'minimum_history_length': 8,    # 3.2 seconds
    'prediction_horizon': 12,       # 4.8 seconds (at least as far as the loss function is concerned)

    ### Variational Objective
    ## Objective Formulation
    'alpha': 1,
    'k': 3,              # number of samples from z during training
    'k_eval': 50,        # number of samples from z during evaluation
    'use_iwae': False,   # only matters if alpha = 1
    'kl_exact': True,    # relevant only if alpha = 1
    ## KL Annealing/Bounding
    'kl_min': 0.07,
    'kl_weight': 1.0,
    'kl_weight_start': 0.0001,
    'kl_decay_rate': 0.99995,
    'kl_crossover': 8000,
    'kl_sigmoid_divisor': 6,

    ### Network Parameters
    ## RNNs/Summarization
    'rnn_kwargs': {"dropout_keep_prob": 0.75},
    'MLP_dropout_keep_prob': 0.9,
    'rnn_io_dropout_keep_prob': 1.0,
    'enc_rnn_dim_multiple_inputs': 8,
    'enc_rnn_dim_edge': 8,
    'enc_rnn_dim_edge_influence': 8,
    'enc_rnn_dim_history': 32,
    'enc_rnn_dim_future': 32,
    'dec_rnn_dim': 128,
    'dec_GMM_proj_MLP_dims': None,
    'sample_model_during_dec': True,
    'dec_sample_model_prob_start': 0.0,
    'dec_sample_model_prob_final': 0.0,
    'dec_sample_model_prob_crossover': 20000,
    'dec_sample_model_prob_divisor': 6,
    ## q_z_xy (encoder)
    'q_z_xy_MLP_dims': None,
    ## p_z_x (encoder)
    'p_z_x_MLP_dims': 16,
    ## p_y_xz (decoder)
    'fuzz_factor': 0.05,
    'GMM_components': 16,
    'log_sigma_min': -10,
    'log_sigma_max': 10,
    'log_p_yt_xz_max': 50,

    ### Discrete Latent Variable
    'N': 2,
    'K': 5,
    ## Relaxed One-Hot Temperature Annealing
    'tau_init': 2.0,
    'tau_final': 0.001,
    'tau_decay_rate': 0.9999,
    ## Logit Clipping
    'use_z_logit_clipping': False,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 3.0,
    'z_logit_clip_crossover': 8000,
    'z_logit_clip_divisor': 6
}

parser = argparse.ArgumentParser()

parser.add_argument('--sgan_models_path', type=str, default='../ref_impls/SocialGAN-PyTorch/models/sgan-models')
parser.add_argument('--sgan_dset_type', default='test', type=str)


parser.add_argument("--dynamic_edges", help="whether to use dynamic edges or not, options are 'no' and 'yes'",
                    type=str, default='yes')
parser.add_argument("--edge_radius", help="the radius (in meters) within which two nodes will be connected by an edge",
                    type=float, default=1.5)
parser.add_argument("--edge_state_combine_method", help="the method to use for combining edges of the same type",
                    type=str, default=None)
parser.add_argument("--edge_influence_combine_method", help="the method to use for combining edge influences",
                    type=str, default=None)
parser.add_argument('--edge_addition_filter', nargs='+', help="what scaling to use for edges as they're created",
                    type=float, default=[0.25, 0.5, 0.75, 1.0]) # We automatically pad left with 0.0
parser.add_argument('--edge_removal_filter', nargs='+', help="what scaling to use for edges as they're removed",
                    type=float, default=[1.0, 0.0]) # We automatically pad right with 0.0
parser.add_argument('--incl_robot_node', help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument('--num_samples', help='how many times to sample from the model',
                    type=int, default=2000)
parser.add_argument('--num_runs', help='how many scenes to predict per model evaluation',
                    type=int, default=100)

parser.add_argument('--device', help='what device to perform training on',
                    type=str, default='cpu')
parser.add_argument("--eval_device", help="what device to use during evaluation",
                    type=str, default='cpu')

parser.add_argument('--seed', help='manual seed to use, default is 123',
                    type=int, default=123)
args = parser.parse_args()


# 44.72 km/h = 12.42 m/s i.e. that's the max value that a velocity coordinate can be.
max_speed = 12.422222

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


method_names = ['sgan', 'our_full', 'our_most_likely']


def pretty_dataset_name(dataset_name):
    if dataset_name == 'eth':
        return 'ETH - Univ'
    elif dataset_name == 'hotel':
        return 'ETH - Hotel'
    elif dataset_name == 'univ':
        return 'UCY - Univ'
    elif dataset_name == 'zara1':
        return 'UCY - Zara 1'
    elif dataset_name == 'zara2':
        return 'UCY - Zara 2'
    else:
        return dataset_name


def plot_run_trajs(data_precondition, dataset_name,
                   our_preds_most_likely_list, our_preds_list,
                   sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
                   data_ids, t_predicts, random_scene_idxs, num_runs):
    eval_dt = eval_data_dict['dt']

    for run in range(num_runs):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlabel(r'$x$ (m)')
        ax.set_ylabel(r'$y$ (m)')
        sgan_preds = sgan_preds_list[run]
        our_most_likely_preds = our_preds_most_likely_list[run]
        our_full_preds = our_preds_list[run]

        random_scene_idx = random_scene_idxs[run]
        data_id = data_ids[random_scene_idx]
        t_predict = t_predicts[random_scene_idx] - 1

        print(run, data_id, t_predict)

        sgan_plotting_tensors = list()
        for key, value in sgan_preds.items():
            sgan_plotting_tensor = torch.cat([value, torch.full((value.shape[0], 1, 2), np.nan)], dim=1)
            sgan_plotting_tensors.append(sgan_plotting_tensor)

        if len(sgan_plotting_tensors) == 0:
            print('len(sgan_plotting_tensors) == 0 for', run, data_id, t_predict, data_precondition, dataset_name)
            continue

        sgan_plotting_tensor = torch.cat(sgan_plotting_tensors, dim=0).view(-1, 2).cpu().numpy()
        ax.plot(sgan_plotting_tensor[:, 0], sgan_plotting_tensor[:, 1],
                color='#EC8F31', label='Social GAN',
                alpha=0.5, linewidth=0.7)

        # Saving some memory
        del sgan_plotting_tensor
        del sgan_plotting_tensors

        labels_to_use = ['Our Method (Full)', r'Our Method ($z_{best}$)']
        colors_to_use = ['blue', '#1FC287']
        for idx, preds_dict in enumerate([our_full_preds, our_most_likely_preds]):
            our_plotting_tensors = list()
            futures_list = list()
            previous_list = list()
            for key, value in preds_dict.items():
                curr_state_val = eval_inputs[key][data_id, t_predict]
                pred_trajs = torch.from_numpy(plot_utils.integrate_trajectory(value.cpu().numpy(), [0, 1],
                                                                              curr_state_val.cpu().numpy(), [0, 1],
                                                                              eval_dt,
                                                                              output_limit=max_speed,
                                                                              velocity_in=True).astype(np.float32))

                our_plotting_tensor = torch.cat([pred_trajs[:, 0], torch.full((pred_trajs.shape[0], 1, 2), np.nan)], dim=1)
                our_plotting_tensors.append(our_plotting_tensor)

                if idx == 1:
                    run_future = eval_inputs[key][data_id, t_predict+1 : t_predict+1+12, :2]
                    run_previous = eval_inputs[key][data_id, t_predict+1-8 : t_predict+1, :2]
                    futures_list.append(run_future)
                    previous_list.append(run_previous)

            if len(our_plotting_tensors) == 0:
                print('len(our_plotting_tensors) == 0 for', run, data_id, t_predict, data_precondition, dataset_name)
                break

            our_plotting_tensor = torch.cat(our_plotting_tensors, dim=0).view(-1, 2).cpu().numpy()
            ax.plot(our_plotting_tensor[:, 0], our_plotting_tensor[:, 1],
                    color=colors_to_use[idx], label=labels_to_use[idx],
                    alpha=0.5, linewidth=0.7)

            if idx == 1:
                futures_tensor = torch.stack(futures_list, dim=0)
                futures_tensor = torch.cat([futures_tensor, torch.full((futures_tensor.shape[0], 1, 2), np.nan)], dim=1)
                futures_tensor = futures_tensor.view(-1, 2).cpu().numpy()
                futures_tensor[futures_tensor == 0] = np.nan
                ax.plot(futures_tensor[:, 0], futures_tensor[:, 1],
                        color='white', label='Ground Truth',
                        linestyle='--',
                        path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

                previous_tensor = torch.stack(previous_list, dim=0)
                previous_tensor = torch.cat([previous_tensor, torch.full((previous_tensor.shape[0], 1, 2), np.nan)], dim=1)
                previous_tensor = previous_tensor.view(-1, 2).cpu().numpy()
                previous_tensor[previous_tensor == 0] = np.nan
                ax.plot(previous_tensor[:, 0], previous_tensor[:, 1],
                        color='k', label='History', linestyle='--')

                curr_tensor = torch.stack(previous_list, dim=0)[:, -1]
                ax.scatter(curr_tensor[:, 0], curr_tensor[:, 1], s=25,
                           c='grey', linewidths=1, edgecolor='k', zorder=10)

        else:
            # If the loop completed without any breaks, we can go ahead
            # and plot the results.
            ax.legend(loc='best')
            plt.savefig('../sgan-dataset/plots/simultaneous_plots/checking_%s_%s_pred_fig_run%d.pdf' % (data_precondition, dataset_name, run), dpi=300, bbox_inches='tight')


def get_kde_log_likelihoods(data_precondition, dataset_name,
                            our_preds_most_likely_list, our_preds_list,
                            sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
                            data_ids, t_predicts, random_scene_idxs, num_runs):
    eval_dt = eval_data_dict['dt']

    all_methods_preds_dict = defaultdict(list)
    gt_dicts = list()
    for run in range(num_runs):
        sgan_preds = sgan_preds_list[run]
        our_most_likely_preds = our_preds_most_likely_list[run]
        our_full_preds = our_preds_list[run]

        random_scene_idx = random_scene_idxs[run]
        data_id = data_ids[random_scene_idx]
        t_predict = t_predicts[random_scene_idx] - 1

        print(run, data_id, t_predict)

        sgan_preds = {key: value.cpu().numpy() for key, value in sgan_preds.items()}
        all_methods_preds_dict['sgan'].append(sgan_preds)

        methods_list = ['our_full', 'our_most_likely']
        curr_gt = dict()
        for idx, preds_dict in enumerate([our_full_preds, our_most_likely_preds]):
            curr_preds = dict()
            for key, value in preds_dict.items():
                curr_state_val = eval_inputs[key][data_id, t_predict]
                pred_trajs = plot_utils.integrate_trajectory(value.cpu().numpy(), [0, 1],
                                                             curr_state_val.cpu().numpy(), [0, 1],
                                                             eval_dt,
                                                             output_limit=max_speed,
                                                             velocity_in=True).astype(np.float32)

                curr_preds[key] = pred_trajs[:, 0]

                if idx == 1:
                    curr_gt[key] = eval_inputs[key][[data_id], t_predict+1 : t_predict+1+12, :2].cpu().numpy()

            all_methods_preds_dict[methods_list[idx]].append(curr_preds)

        gt_dicts.append(curr_gt)

    detailed_ll_dict = {'data_precondition': list(),
                        'dataset': list(),
                        'method': list(),
                        'run': list(),
                        'timestep': list(),
                        'node': list(),
                        'log-likelihood': list()}

    sgan_lls = list()
    our_full_lls = list()
    our_most_likely_lls = list()
    log_pdf_lower_bound = -20
    for run in range(num_runs):
        sgan_preds = all_methods_preds_dict['sgan'][run]
        our_full_preds = all_methods_preds_dict['our_full'][run]
        our_most_likely_preds = all_methods_preds_dict['our_most_likely'][run]
        gt_dict = gt_dicts[run]

        for node in sgan_preds.keys():
            first_nz = plot_utils.first_nonzero(np.sum(gt_dict[node], axis=2)[0, ::-1], axis=0)
            if first_nz < 0:
                continue
            num_timesteps = gt_dict[node].shape[1] - first_nz

            sgan_ll = 0.0
            our_full_ll = 0.0
            our_most_likely_ll = 0.0
            for timestep in range(num_timesteps):
                curr_gt = gt_dict[node][:, timestep]

                sgan_scipy_kde = gaussian_kde(sgan_preds[node][:, timestep].T)
                our_full_scipy_kde = gaussian_kde(our_full_preds[node][:, timestep].T)
                our_most_likely_scipy_kde = gaussian_kde(our_most_likely_preds[node][:, timestep].T)

                # We need [0] because it's a (1,)-shaped numpy array.
                sgan_log_pdf = np.clip(sgan_scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
                our_full_pdf = np.clip(our_full_scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
                our_most_likely_pdf = np.clip(our_most_likely_scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]

                for idx, result in enumerate([sgan_log_pdf, our_full_pdf, our_most_likely_pdf]):
                    detailed_ll_dict['data_precondition'].append(data_precondition)
                    detailed_ll_dict['dataset'].append(dataset_name)
                    detailed_ll_dict['method'].append(method_names[idx])
                    detailed_ll_dict['run'].append(run)
                    detailed_ll_dict['timestep'].append(timestep)
                    detailed_ll_dict['node'].append(str(node))
                    detailed_ll_dict['log-likelihood'].append(result)

                sgan_ll += sgan_log_pdf/num_timesteps
                our_full_ll += our_full_pdf/num_timesteps
                our_most_likely_ll += our_most_likely_pdf/num_timesteps

            sgan_lls.append(sgan_ll)
            our_full_lls.append(our_full_ll)
            our_most_likely_lls.append(our_most_likely_ll)

    return sgan_lls, our_full_lls, our_most_likely_lls, detailed_ll_dict


def main():
    for data_precondition in ['curr', 'prev', 'all']:
        sgan_eval_mse_batch_errors = dict()
        sgan_eval_fse_batch_errors = dict()
        our_eval_mse_batch_errors = dict()
        our_eval_fse_batch_errors = dict()
        for dataset_name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
            (our_preds_most_likely_list, our_preds_list,
             sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
             data_ids, t_predicts, random_scene_idxs, num_runs) = eval_utils.extract_our_and_sgan_preds(dataset_name, hyperparams, args, data_precondition=data_precondition)

            # Computing log-likelihoods from each run.
            sgan_lls, our_full_lls, our_most_likely_lls, detailed_ll_dict = get_kde_log_likelihoods(
                data_precondition, dataset_name,
                our_preds_most_likely_list, our_preds_list,
                sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
                data_ids, t_predicts, random_scene_idxs, num_runs)
            print('SGAN LLs, Our Method (Full) LLs, Our Method (Most Likely) LLs')
            print(np.mean(sgan_lls), np.mean(our_full_lls), np.mean(our_most_likely_lls))
            print('Calculated all KDE LLs for', data_precondition, dataset_name)

            print('Saving current log-likelihoods to csv.')
            pd.DataFrame.from_dict(detailed_ll_dict).to_csv('../sgan-dataset/plots/data/%s_%s_lls.csv' % (data_precondition, dataset_name), index=False)

            # Plotting the trajectories from each run.
            plot_run_trajs(data_precondition, dataset_name,
                           our_preds_most_likely_list, our_preds_list,
                           sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
                           data_ids, t_predicts, random_scene_idxs, num_runs)
            print('Plotted all run trajectories from', data_precondition, dataset_name)

            # SGAN Errors
            batch_error_dict, detailed_error_dict = eval_utils.compute_sgan_errors(
                                                            sgan_preds_list,
                                                            sgan_gt_list,
                                                            data_precondition,
                                                            dataset_name,
                                                            num_runs)

            print('Saving current SGAN errors to csv.')
            pd.DataFrame.from_dict(detailed_error_dict).to_csv('../sgan-dataset/plots/data/%s_%s_sgan_errors.csv' % (data_precondition, dataset_name), index=False)

            sgan_eval_mse_batch_errors[pretty_dataset_name(dataset_name)] = torch.cat(batch_error_dict['mse'], dim=0)
            sgan_eval_fse_batch_errors[pretty_dataset_name(dataset_name)] = torch.cat(batch_error_dict['fse'], dim=0)

            # Our Most Likely Errors
            error_info_dict = {'output_limit': max_speed}
            batch_error_dict, detailed_error_dict = eval_utils.compute_preds_dict_error(
                                            our_preds_most_likely_list,
                                            eval_data_dict,
                                            data_precondition,
                                            dataset_name,
                                            'our_most_likely',
                                            num_runs,
                                            random_scene_idxs,
                                            data_ids,
                                            t_predicts,
                                            hyperparams['prediction_horizon'],
                                            error_info_dict)

            print('Saving current Our Method (Most Likely) errors to csv.')
            pd.DataFrame.from_dict(detailed_error_dict).to_csv('../sgan-dataset/plots/data/%s_%s_our_most_likely_errors.csv' % (data_precondition, dataset_name), index=False)

            our_eval_mse_batch_errors[pretty_dataset_name(dataset_name)] = torch.cat(batch_error_dict['mse'], dim=0)
            our_eval_fse_batch_errors[pretty_dataset_name(dataset_name)] = torch.cat(batch_error_dict['fse'], dim=0)

            # Our Full Errors
            error_info_dict = {'output_limit': max_speed}
            batch_error_dict, detailed_error_dict = eval_utils.compute_preds_dict_error(
                                            our_preds_list,
                                            eval_data_dict,
                                            data_precondition,
                                            dataset_name,
                                            'our_full',
                                            num_runs,
                                            random_scene_idxs,
                                            data_ids,
                                            t_predicts,
                                            hyperparams['prediction_horizon'],
                                            error_info_dict)

            print('Saving current Our Method (Full) errors to csv.')
            pd.DataFrame.from_dict(detailed_error_dict).to_csv('../sgan-dataset/plots/data/%s_%s_our_full_errors.csv' % (data_precondition, dataset_name), index=False)


        with torch.no_grad():
            mse_boxplot_fig, fse_boxplot_fig = plot_utils.plot_comparison_boxplots(
                                                    sgan_eval_mse_batch_errors,
                                                    sgan_eval_fse_batch_errors,
                                                    our_eval_mse_batch_errors,
                                                    our_eval_fse_batch_errors)

        mse_boxplot_fig.savefig('../sgan-dataset/plots/%s_comparison_mse_boxplot.pdf' % data_precondition, dpi=300, bbox_inches="tight")
        fse_boxplot_fig.savefig('../sgan-dataset/plots/%s_comparison_fse_boxplot.pdf' % data_precondition, dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
