import torch
import numpy as np
import pandas as pd
from stg_node import STGNode, convert_to_label_node
from utils import plot_utils
import os
import time
import copy
import pickle
import argparse
import random
from collections import defaultdict, Counter

from model.online_dyn_stg import OnlineSpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from utils.scene_utils import SceneGraph
from utils import eval_utils

from attrdict import AttrDict
import sys
sys.path.append('../ref_impls/SocialGAN-PyTorch')

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, get_dset_path


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
                    type=int, default=200)
parser.add_argument('--num_runs', help='how many scenes to predict per model evaluation',
                    type=int, default=10)

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


def main():
    results_dict = {'data_precondition': list(),
                    'dataset': list(),
                    'method': list(),
                    'runtime': list(),
                    'num_samples': list(),
                    'num_agents': list()}
    data_precondition = 'curr'
    for dataset_name in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
        print('At %s dataset' % dataset_name)

        ### SGAN LOADING ###
        sgan_model_path = os.path.join(args.sgan_models_path, '_'.join([dataset_name, '12', 'model.pt']))

        checkpoint = torch.load(sgan_model_path, map_location='cpu')
        generator = eval_utils.get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.sgan_dset_type)
        print('Evaluating', sgan_model_path, 'on', _args.dataset_name, args.sgan_dset_type)

        _, sgan_data_loader = data_loader(_args, path)

        ### OUR METHOD LOADING ###
        data_dir = '../sgan-dataset/data'
        eval_data_dict_name = '%s_test.pkl' % dataset_name
        log_dir = '../sgan-dataset/logs/%s' % dataset_name

        trained_model_dir = os.path.join(log_dir, eval_utils.get_our_model_dir(dataset_name))
        eval_data_path = os.path.join(data_dir, eval_data_dict_name)
        with open(eval_data_path, 'rb') as f:
            eval_data_dict = pickle.load(f, encoding='latin1')
        eval_dt = eval_data_dict['dt']
        print('Loaded evaluation data from %s, eval_dt = %.2f' % (eval_data_path, eval_dt))

        # Loading weights from the trained model.
        specific_hyperparams = eval_utils.get_model_hyperparams(args, dataset_name)
        model_registrar = ModelRegistrar(trained_model_dir, args.device)
        model_registrar.load_models(specific_hyperparams['best_iter'])

        for key in eval_data_dict['input_dict'].keys():
            if isinstance(key, STGNode):
                random_node = key
                break

        hyperparams['state_dim'] = eval_data_dict['input_dict'][random_node].shape[2]
        hyperparams['pred_dim'] = len(eval_data_dict['pred_indices'])
        hyperparams['pred_indices'] = eval_data_dict['pred_indices']
        hyperparams['dynamic_edges'] = args.dynamic_edges
        hyperparams['edge_state_combine_method'] = specific_hyperparams['edge_state_combine_method']
        hyperparams['edge_influence_combine_method'] = specific_hyperparams['edge_influence_combine_method']
        hyperparams['nodes_standardization'] = eval_data_dict['nodes_standardization']
        hyperparams['labels_standardization'] = eval_data_dict['labels_standardization']
        hyperparams['edge_radius'] = args.edge_radius

        eval_hyperparams = copy.deepcopy(hyperparams)
        eval_hyperparams['nodes_standardization'] = eval_data_dict["nodes_standardization"]
        eval_hyperparams['labels_standardization'] = eval_data_dict["labels_standardization"]

        kwargs_dict = {'dynamic_edges': hyperparams['dynamic_edges'],
                       'edge_state_combine_method': hyperparams['edge_state_combine_method'],
                       'edge_influence_combine_method': hyperparams['edge_influence_combine_method'],
                       'edge_addition_filter': args.edge_addition_filter,
                       'edge_removal_filter': args.edge_removal_filter}

        print('-------------------------')
        print('| EVALUATION PARAMETERS |')
        print('-------------------------')
        print('| checking: %s' % data_precondition)
        print('| device: %s' % args.device)
        print('| eval_device: %s' % args.eval_device)
        print('| edge_radius: %s' % hyperparams['edge_radius'])
        print('| EE state_combine_method: %s' % hyperparams['edge_state_combine_method'])
        print('| EIE scheme: %s' % hyperparams['edge_influence_combine_method'])
        print('| dynamic_edges: %s' % hyperparams['dynamic_edges'])
        print('| edge_addition_filter: %s' % args.edge_addition_filter)
        print('| edge_removal_filter: %s' % args.edge_removal_filter)
        print('| MHL: %s' % hyperparams['minimum_history_length'])
        print('| PH: %s' % hyperparams['prediction_horizon'])
        print('| # Samples: %s' % args.num_samples)
        print('| # Runs: %s' % args.num_runs)
        print('-------------------------')

        eval_stg = OnlineSpatioTemporalGraphCVAEModel(None, model_registrar,
                                                eval_hyperparams, kwargs_dict,
                                                args.eval_device)
        print('Created evaluation STG model.')

        print('About to begin evaluation computation for %s.' % dataset_name)
        with torch.no_grad():
            eval_inputs, _ = eval_utils.sample_inputs_and_labels(eval_data_dict, device=args.eval_device)

        (obs_traj, pred_traj_gt, obs_traj_rel,
         seq_start_end, data_ids, t_predicts) = eval_utils.get_sgan_data_format(eval_inputs, what_to_check=data_precondition)

        num_runs = args.num_runs
        print('num_runs, seq_start_end.shape[0]', args.num_runs, seq_start_end.shape[0])
        if args.num_runs > seq_start_end.shape[0]:
            print('num_runs (%d) > seq_start_end.shape[0] (%d), reducing num_runs to match.' % (num_runs, seq_start_end.shape[0]))
            num_runs = seq_start_end.shape[0]

        random_scene_idxs = np.random.choice(seq_start_end.shape[0],
                                             size=(num_runs,),
                                             replace=False).astype(int)

        for scene_idxs in random_scene_idxs:
            choice_list = seq_start_end[scene_idxs]

            overall_tic = time.time()
            for sample_num in range(args.num_samples):
                pred_traj_fake_rel = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

            overall_toc = time.time()
            print('SGAN overall', overall_toc - overall_tic)
            results_dict['data_precondition'].append(data_precondition)
            results_dict['dataset'].append(dataset_name)
            results_dict['method'].append('sgan')
            results_dict['runtime'].append(overall_toc - overall_tic)
            results_dict['num_samples'].append(args.num_samples)
            results_dict['num_agents'].append(int(choice_list[1].item() - choice_list[0].item()))

        print('Done running SGAN')

        for node in eval_data_dict['nodes_standardization']:
            for key in eval_data_dict['nodes_standardization'][node]:
                eval_data_dict['nodes_standardization'][node][key] = torch.from_numpy(eval_data_dict['nodes_standardization'][node][key]).float().to(args.device)

        for node in eval_data_dict['labels_standardization']:
            for key in eval_data_dict['labels_standardization'][node]:
                eval_data_dict['labels_standardization'][node][key] = torch.from_numpy(eval_data_dict['labels_standardization'][node][key]).float().to(args.device)

        for run in range(num_runs):
            random_scene_idx = random_scene_idxs[run]
            data_id = data_ids[random_scene_idx]
            t_predict = t_predicts[random_scene_idx] - 1

            init_scene_dict = dict()
            for first_timestep in range(t_predict+1):
                for node, traj_data in eval_data_dict['input_dict'].items():
                    if isinstance(node, STGNode):
                        init_pos = traj_data[data_id, first_timestep, :2]
                        if np.any(init_pos):
                            init_scene_dict[node] = init_pos

                if len(init_scene_dict) > 0:
                    break

            init_scene_graph = SceneGraph()
            init_scene_graph.create_from_scene_dict(init_scene_dict, args.edge_radius)

            curr_inputs = {k: v[data_id, first_timestep:t_predict+1] for k, v in eval_data_dict['input_dict'].items() if (isinstance(k, STGNode) and (k in init_scene_graph.active_nodes))}
            curr_pos_inputs = {k: v[..., :2] for k, v in curr_inputs.items()}

            with torch.no_grad():
                overall_tic = time.time()
                preds_dict_most_likely = eval_stg.forward(init_scene_graph,
                                                          curr_pos_inputs,
                                                          curr_inputs,
                                                          None,
                                                          hyperparams['prediction_horizon'],
                                                          args.num_samples,
                                                          most_likely=True)
                overall_toc = time.time()
                print('Our MLz overall', overall_toc - overall_tic)
                results_dict['data_precondition'].append(data_precondition)
                results_dict['dataset'].append(dataset_name)
                results_dict['method'].append('our_most_likely')
                results_dict['runtime'].append(overall_toc - overall_tic)
                results_dict['num_samples'].append(args.num_samples)
                results_dict['num_agents'].append(len(init_scene_dict))

                overall_tic = time.time()
                preds_dict_full = eval_stg.forward(init_scene_graph,
                                                   curr_pos_inputs,
                                                   curr_inputs,
                                                   None,
                                                   hyperparams['prediction_horizon'],
                                                   args.num_samples,
                                                   most_likely=False)
                overall_toc = time.time()
                print('Our Full overall', overall_toc - overall_tic)
                results_dict['data_precondition'].append(data_precondition)
                results_dict['dataset'].append(dataset_name)
                results_dict['method'].append('our_full')
                results_dict['runtime'].append(overall_toc - overall_tic)
                results_dict['num_samples'].append(args.num_samples)
                results_dict['num_agents'].append(len(init_scene_dict))

        pd.DataFrame.from_dict(results_dict).to_csv('../sgan-dataset/plots/data/%s_%s_runtimes.csv' % (data_precondition, dataset_name), index=False)


if __name__ == '__main__':
    main()
