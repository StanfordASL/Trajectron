import torch
import numpy as np
import os

import timeit
import matplotlib
matplotlib.use('Agg');
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from PIL import Image
import random
from collections import defaultdict, OrderedDict
import matplotlib.patheffects as pe

import pandas as pd
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW


# These are for a prediction horizon of 12 timesteps.
prior_work_mse_results = {
    'ETH - Univ': OrderedDict([('Linear', 1.33), ('Vanilla LSTM', 1.09), ('Social LSTM', 1.09), ('Social Attention', 0.39)]),
    'ETH - Hotel': OrderedDict([('Linear', 0.39), ('Vanilla LSTM', 0.86), ('Social LSTM', 0.79), ('Social Attention', 0.29)]),
    'UCY - Univ': OrderedDict([('Linear', 0.82), ('Vanilla LSTM', 0.61), ('Social LSTM', 0.67), ('Social Attention', 0.20)]),
    'UCY - Zara 1': OrderedDict([('Linear', 0.62), ('Vanilla LSTM', 0.41), ('Social LSTM', 0.47), ('Social Attention', 0.30)]),
    'UCY - Zara 2': OrderedDict([('Linear', 0.77), ('Vanilla LSTM', 0.52), ('Social LSTM', 0.56), ('Social Attention', 0.33)]),
    'Average': OrderedDict([('Linear', 0.79), ('Vanilla LSTM', 0.70), ('Social LSTM', 0.72), ('Social Attention', 0.30)])
}


prior_work_fse_results = {
    'ETH - Univ': OrderedDict([('Linear', 2.94), ('Vanilla LSTM', 2.41), ('Social LSTM', 2.35), ('Social Attention', 3.74)]),
    'ETH - Hotel': OrderedDict([('Linear', 0.72), ('Vanilla LSTM', 1.91), ('Social LSTM', 1.76), ('Social Attention', 2.64)]),
    'UCY - Univ': OrderedDict([('Linear', 1.59), ('Vanilla LSTM', 1.31), ('Social LSTM', 1.40), ('Social Attention', 0.52)]),
    'UCY - Zara 1': OrderedDict([('Linear', 1.21), ('Vanilla LSTM', 0.88), ('Social LSTM', 1.00), ('Social Attention', 2.13)]),
    'UCY - Zara 2': OrderedDict([('Linear', 1.48), ('Vanilla LSTM', 1.11), ('Social LSTM', 1.17), ('Social Attention', 3.92)]),
    'Average': OrderedDict([('Linear', 1.59), ('Vanilla LSTM', 1.52), ('Social LSTM', 1.54), ('Social Attention', 2.59)])
}

linestyles = ['--', '-.', '-', ':']


def plot_hyperparam_boxplots(our_mse_errors, our_fse_errors):
    perf_dict_for_pd = {'run': list(), 'mse': list(), 'fse': list()}
    for key in our_mse_errors:
        if key in our_fse_errors:
            perf_dict_for_pd['run'].extend([key]*(our_mse_errors[key].shape[0]))
            perf_dict_for_pd['mse'].extend(our_mse_errors[key].numpy().tolist())
            perf_dict_for_pd['fse'].extend(our_fse_errors[key].numpy().tolist())

    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    datasets_in_perf_df = pd.unique(perf_df['run'])

    line_colors = ['#1f78b4','#33a02c','#fb9a99','#e31a1c']
    area_colors = ['#a6cee3','#b2df8a']
    with sns.color_palette("muted"):
        fig_mse, ax_mses = plt.subplots(nrows=1, ncols=len(datasets_in_perf_df), figsize=(10, 5), sharey=True)
        for idx, ax_mse in enumerate(ax_mses):
            dataset_name = datasets_in_perf_df[idx]
            specific_df = perf_df[perf_df['run'] == dataset_name]

            sns.boxplot(x='run', y='mse',
                        data=specific_df, ax=ax_mse, showfliers=False,
                        palette=area_colors)

            ax_mse.set_xlabel('')
            ax_mse.set_ylabel('' if idx > 0 else 'Average Displacement Error (m)')

            # ax_mse.plot([-0.2, 0.2],
            #                [np.mean(specific_df[specific_df['approach'] == 'Social GAN']['mse']),
            #                 np.mean(specific_df[specific_df['approach'] == 'Our Method']['mse'])],
            #                linestyle='None',
            #                color=mean_color, marker=mean_markers,
            #                markeredgecolor='#545454', markersize=marker_size, zorder=10)

        fig_mse.text(0.51, 0.03, 'Run', ha='center')

        fig_fse, ax_fses = plt.subplots(nrows=1, ncols=len(datasets_in_perf_df), figsize=(10, 5), sharey=True)
        for idx, ax_fse in enumerate(ax_fses):
            dataset_name = datasets_in_perf_df[idx]
            specific_df = perf_df[perf_df['run'] == dataset_name]

            sns.boxplot(x='run', y='fse',
                        data=specific_df, ax=ax_fse, showfliers=False,
                        palette=area_colors)

            ax_fse.set_xlabel('')
            ax_fse.set_ylabel('' if idx > 0 else 'Final Displacement Error (m)')

            # ax_fse.plot([-0.2, 0.2],
            #                [np.mean(perf_df[perf_df['approach'] == 'Social GAN']['fse']),
            #                 np.mean(perf_df[perf_df['approach'] == 'Our Method']['fse'])],
            #                linestyle='None',
            #                color=mean_color, marker=mean_markers,
            #                markeredgecolor='#545454', markersize=marker_size, zorder=10)


        fig_fse.text(0.51, 0.03, 'Run', ha='center')

        return fig_mse, fig_fse


def plot_comparison_boxplots(their_mse_errors, their_fse_errors,
                             our_mse_errors, our_fse_errors):
    dataset_names = ['ETH - Univ', 'ETH - Hotel', 'UCY - Univ', 'UCY - Zara 1', 'UCY - Zara 2']
    perf_dict_for_pd = {'dataset': list(), 'approach': list(), 'mse': list(), 'fse': list()}
    for dataset_name in dataset_names:
        if dataset_name in their_mse_errors and dataset_name in their_fse_errors:
            perf_dict_for_pd['dataset'].extend([dataset_name]*(their_mse_errors[dataset_name].shape[0]))
            perf_dict_for_pd['approach'].extend(['Social GAN']*their_mse_errors[dataset_name].shape[0])
            perf_dict_for_pd['mse'].extend(their_mse_errors[dataset_name].numpy().tolist())
            perf_dict_for_pd['fse'].extend(their_fse_errors[dataset_name].numpy().tolist())

        if dataset_name in our_mse_errors and dataset_name in our_fse_errors:
            perf_dict_for_pd['dataset'].extend([dataset_name]*(our_mse_errors[dataset_name].shape[0]))
            perf_dict_for_pd['approach'].extend(['Our Method']*our_mse_errors[dataset_name].shape[0])
            perf_dict_for_pd['mse'].extend(our_mse_errors[dataset_name].numpy().tolist())
            perf_dict_for_pd['fse'].extend(our_fse_errors[dataset_name].numpy().tolist())

    dataset_names += ['Average']

    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    datasets_in_perf_df = pd.unique(perf_df['dataset'])

    line_colors = ['#1f78b4','#33a02c','#fb9a99','#e31a1c']
    area_colors = ['#a6cee3','#b2df8a']
    area_rgbs = list()
    for c in area_colors:
        area_rgbs.append([int(c[i:i+2], 16) for i in (1, 3, 5)])

    mean_markers = 'X'
    marker_size = 7
    with sns.color_palette("muted"):
        fig_mse, ax_mses = plt.subplots(nrows=1, ncols=6, figsize=(10, 5), sharey=True)
        for idx, ax_mse in enumerate(ax_mses):
            dataset_name = dataset_names[idx]
            if dataset_name != 'Average' and dataset_name in datasets_in_perf_df:
                specific_df = perf_df[perf_df['dataset'] == dataset_name]
            elif dataset_name == 'Average':
                specific_df = perf_df.copy()
                specific_df['dataset'] = 'Average'
            else:
                print('No data found for %s, skipping!' % dataset_name)
                continue

            for baseline_idx, (baseline, mse_val) in enumerate(prior_work_mse_results[dataset_name].items()):
                ax_mse.axhline(y=mse_val, label=baseline, color=line_colors[baseline_idx], linestyle=linestyles[baseline_idx])

            sns.boxplot(x='dataset', y='mse', hue='approach',
                        data=specific_df, ax=ax_mse, showfliers=False,
                        palette=area_colors)

            ax_mse.get_legend().remove()
            ax_mse.set_xlabel('')
            ax_mse.set_ylabel('' if idx > 0 else 'Average Displacement Error (m)')

            if idx == 0:
                ax_mse.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9),
                              ncol=6, borderaxespad=0, frameon=False,
                              bbox_transform=fig_mse.transFigure)

            ax_mse.scatter([-0.2, 0.2],
                           [np.mean(specific_df[specific_df['approach'] == 'Social GAN']['mse']),
                            np.mean(specific_df[specific_df['approach'] == 'Our Method']['mse'])],
                           s=marker_size*marker_size, c=np.asarray(area_rgbs)/255.0, marker=mean_markers,
                           edgecolors='#545454', zorder=10)

        fig_mse.text(0.51, 0.03, 'Dataset', ha='center')

        fig_fse, ax_fses = plt.subplots(nrows=1, ncols=6, figsize=(10, 5), sharey=True)
        for idx, ax_fse in enumerate(ax_fses):
            dataset_name = dataset_names[idx]
            if dataset_name != 'Average' and dataset_name in datasets_in_perf_df:
                specific_df = perf_df[perf_df['dataset'] == dataset_name]
            elif dataset_name == 'Average':
                specific_df = perf_df.copy()
                specific_df['dataset'] = 'Average'
            else:
                print('No data found for %s, skipping!' % dataset_name)
                continue

            for baseline_idx, (baseline, fse_val) in enumerate(prior_work_fse_results[dataset_name].items()):
                ax_fse.axhline(y=fse_val, label=baseline, color=line_colors[baseline_idx], linestyle=linestyles[baseline_idx])

            sns.boxplot(x='dataset', y='fse', hue='approach',
                        data=specific_df, ax=ax_fse, showfliers=False,
                        palette=area_colors)

            ax_fse.get_legend().remove()
            ax_fse.set_xlabel('')
            ax_fse.set_ylabel('' if idx > 0 else 'Final Displacement Error (m)')

            if idx == 0:
                ax_fse.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9),
                              ncol=6, borderaxespad=0, frameon=False,
                              bbox_transform=fig_fse.transFigure)

            ax_fse.scatter([-0.2, 0.2],
                           [np.mean(specific_df[specific_df['approach'] == 'Social GAN']['fse']),
                            np.mean(specific_df[specific_df['approach'] == 'Our Method']['fse'])],
                           s=marker_size*marker_size, c=np.asarray(area_rgbs)/255.0, marker=mean_markers,
                           edgecolors='#545454', zorder=10)


        fig_fse.text(0.51, 0.03, 'Dataset', ha='center')

        return fig_mse, fig_fse


def plot_boxplots_during_training(train_mse_batch_errors, train_fse_batch_errors):
    perf_dict_for_pd = {'dataset': ['ETH']*train_mse_batch_errors.shape[0],
                        'mse': train_mse_batch_errors.numpy().tolist(),
                        'fse': train_fse_batch_errors.numpy().tolist()}

    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)
    our_mean_color = sns.color_palette("muted")[9]
    marker_size = 7
    mean_markers = 'X'
    with sns.color_palette("muted"):
        fig_mse, ax_mse = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='dataset', y='mse', data=perf_df, ax=ax_mse, showfliers=False)
        ax_mse.plot([0], [np.mean(perf_df['mse'])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)
        ax_mse.set_ylabel('Average Displacement Error')
        fig_mse.tight_layout()

        fig_fse, ax_fse = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='dataset', y='fse', data=perf_df, ax=ax_fse, showfliers=False)
        ax_fse.plot([0], [np.mean(perf_df['fse'])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)
        ax_fse.set_ylabel('Final Displacement Error')
        fig_fse.tight_layout()

        return fig_mse, fig_fse


def plot_performance_metrics(perf_dict, output_save_dir, minimum_history_length, prediction_horizon):
    perf_dict_for_pd = {'dataset': list(), 'mse': list(), 'fse': list()}
    for idx in range(minimum_history_length, len(perf_dict['time'])):
        curr_mse_arr = perf_dict['mse'][idx].cpu().numpy()
        curr_fse_arr = perf_dict['fse'][idx].cpu().numpy()
        for err_idx in range(curr_mse_arr.shape[0]):
            perf_dict_for_pd['dataset'].append('ETH')
            perf_dict_for_pd['mse'].append(curr_mse_arr[err_idx])
            perf_dict_for_pd['fse'].append(curr_fse_arr[err_idx])

    perf_df = pd.DataFrame.from_dict(perf_dict_for_pd)

    sota_mean_color = sns.color_palette("muted")[2]
    our_mean_color = sns.color_palette("muted")[9]
    marker_size = 7
    mean_markers = 'X'
    with sns.color_palette("muted"):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='dataset', y='mse', data=perf_df, ax=ax, showfliers=False)
        ax.plot([0], [np.mean(perf_df['mse'])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)
        # ax.plot([0], [0.30], color=sota_mean_color, marker=mean_markers,
        #         markeredgecolor='#545454', markersize=marker_size)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Average Displacement Error')
        plt.tight_layout()
        plt.savefig(os.path.join(output_save_dir, 'mse_for_run.pdf'), dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(x='dataset', y='fse', data=perf_df, ax=ax, showfliers=False)
        ax.plot([0], [np.mean(perf_df['fse'])], color=our_mean_color, marker=mean_markers,
                markeredgecolor='#545454', markersize=marker_size, zorder=10)
        # ax.plot([0], [2.59], color=sota_mean_color, marker=mean_markers,
        #         markeredgecolor='#545454', markersize=marker_size)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Final Displacement Error')
        plt.tight_layout()
        plt.savefig(os.path.join(output_save_dir, 'fse_for_run.pdf'), dpi=300)
        plt.close(fig)

    mse_descriptive_stats = DescrStatsW(perf_df['mse'])
    print('MSE Mean and 95% CI:', mse_descriptive_stats.mean, mse_descriptive_stats.tconfint_mean())
    # print('Two-sided t-test vs. 0.30:', mse_descriptive_stats.ttest_mean(0.30))

    fse_descriptive_stats = DescrStatsW(perf_df['fse'])
    print('FSE Mean and 95% CI:', fse_descriptive_stats.mean, fse_descriptive_stats.tconfint_mean())
    # print('Two-sided t-test vs. 2.59:', fse_descriptive_stats.ttest_mean(2.59))


def integrate_trajectory(input_traj, input_dims,
                         curr_state, curr_state_dims, dt,
                         vel_dims=[2, 3], output_limit=None, velocity_in=False):
    if velocity_in and output_limit is not None:
        # Ensuring vector magnitude is <= output_limit
        integrated = input_traj[..., input_dims]
        offending_indices = np.linalg.norm(integrated, axis=3) > output_limit
        num_offending = np.count_nonzero(offending_indices)

        if num_offending > 0:
            # print('WARNING! Had %d offending inputs!' % num_offending)
            dividing_factor = np.linalg.norm(integrated[offending_indices], axis=1, keepdims=True) / output_limit
            integrated[offending_indices] = integrated[offending_indices] / dividing_factor

        input_traj[..., input_dims] = integrated

    # Extending by one the time horizon (via duplication of the last point)
    # because of trapezoidal integration taking a timestep off the end.
    extended_input_traj = np.concatenate([input_traj, input_traj[:, :, [-1]]], axis=2)
    xd_ = cumtrapz(extended_input_traj[..., input_dims[0]], axis=2, dx=dt) + curr_state[curr_state_dims[0]]
    yd_ = cumtrapz(extended_input_traj[..., input_dims[1]], axis=2, dx=dt) + curr_state[curr_state_dims[1]]
    integrated = np.stack([xd_, yd_], axis=3)

    if not velocity_in and output_limit is not None:
        # Ensuring vector magnitude is <= output_limit
        offending_indices = np.linalg.norm(integrated, axis=3) > output_limit
        num_offending = np.count_nonzero(offending_indices)

        if num_offending > 0:
            # print('WARNING! Had %d offending inputs!' % num_offending)
            dividing_factor = np.linalg.norm(integrated[offending_indices], axis=1, keepdims=True) / output_limit
            integrated[offending_indices] = integrated[offending_indices] / dividing_factor

    return integrated


def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def plot_predictions_during_training(test_stg, inputs, num_predicted_timesteps,
                                     num_samples, dt, max_speed,
                                     color_dict=None, most_likely=False,
                                     data_id=None, t_predict=56,
                                     focus_on=None, node_circle_size=0.3,
                                     focus_window_height=6, line_alpha=0.7,
                                     line_width=0.2, edge_width=2, circle_edge_width=0.5,
                                     only_predict=None,
                                     dpi=300, fig_height=4,
                                     return_frame=False, return_fig=True,
                                     return_preds=False,
                                     printing=False, robot_circle=True,
                                     add_legend=True, title=None,
                                     flip_axes=False, omit_names=False,
                                     axes_labels=True, rotate_axes_text=0,
                                     save_at=None):

    if color_dict is None:
        # This changes colors per run.
        color_dict = defaultdict(dict)

    if dt is None:
        raise ValueError('You must supply a time delta, the argument dt cannot be None!')

    robot_node = test_stg.robot_node
    figsize = (5, 5)
    traj_lengths = inputs['traj_lengths'].cpu().numpy().astype(int)
    predict_horizon = num_predicted_timesteps

    if data_id is None:
        data_id = np.random.choice([idx for idx, traj_len in enumerate(traj_lengths) if t_predict + predict_horizon < traj_len])

    for key in inputs.keys():
        if len(inputs[key].shape) > 2:
            random_node = key

    max_time = inputs[random_node][data_id, :].shape[0]
    traj_length = traj_lengths[data_id]

    if t_predict < test_stg.hyperparams['minimum_history_length']:
        raise ValueError('ERROR: t_predict must be >= %d' % test_stg.hyperparams['minimum_history_length'])
    elif t_predict + predict_horizon >= traj_length:
        raise ValueError('ERROR: t_predict must be <= %d' % (traj_length - predict_horizon - 1))

    ###################
    ### Predictions ###
    ###################
    inputs = {k: v[[data_id]] for k, v in inputs.items()}
    if robot_node is not None:
        robot_future = inputs[robot_node][[0], t_predict + 1 : t_predict + predict_horizon + 1]
        inputs[str(robot_node) + "_future"] = robot_future

    inputs['traj_lengths'] = torch.tensor([t_predict])

    with torch.no_grad():
        tic0 = timeit.default_timer()
        outputs = test_stg.predict(inputs, num_predicted_timesteps, num_samples, most_likely=most_likely)
        toc0 = timeit.default_timer()

        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

    if printing:
        print("done running pytorch!, took (s): ", toc0 - tic0)

    ########################
    ### Data Preparation ###
    ########################
    prefixes_dict = dict()
    futures_dict = dict()
    nodes = list(test_stg.nodes)
    prefix_earliest_idx = max(0, t_predict - predict_horizon)
    for node in nodes:
        if robot_node == node:
            continue

        node_data = inputs[node].cpu().numpy()
        if printing:
            print('node', node)
            print('node_data.shape', node_data.shape)

        prefixes_dict[node] = node_data[[0], prefix_earliest_idx : t_predict + 1, 0:2]
        futures_dict[node] = node_data[[0], t_predict + 1 : min(t_predict + predict_horizon + 1, traj_length)]
        if printing:
            print('node', node)
            print('prefixes_dict[node].shape', prefixes_dict[node].shape)
            print('futures_dict[node].shape', futures_dict[node].shape)

    # 44.72 km/h = 40.76 ft/s (ie. that's the max value that a coordinate can be)
    output_pos = dict()
    sampled_zs = dict()
    for node in nodes:
        if robot_node == node:
            continue

        key = str(node) + '/y'
        z_key = str(node) + '/z'

        output_pos[node] = integrate_trajectory(outputs[key], [0, 1],
                                                prefixes_dict[node][0, -1], [0, 1],
                                                dt, output_limit=max_speed,
                                                velocity_in=True)
        sampled_zs[node] = outputs[z_key]

    if printing:
        print('prefixes_dict[node].shape', prefixes_dict[node].shape)
        print('futures_dict[node].shape', futures_dict[node].shape)
        print('output_pos[node].shape', output_pos[node].shape)
        print('sampled_zs[node].shape', sampled_zs[node].shape)

    ######################
    ### Visualizations ###
    ######################
    fig, ax = plt.subplots(figsize=figsize)
    not_added_prefix = True
    not_added_future = True
    not_added_samples = True
    for node_name in prefixes_dict:
        if focus_on is not None and node_name != focus_on:
            continue

        prefix = prefixes_dict[node_name][0]
        future = futures_dict[node_name][0]
        predictions = output_pos[node_name][:, 0]
        z_values = sampled_zs[node_name][:, 0]

        prefix_all_zeros = not np.any(prefix)
        future_all_zeros = not np.any(future)
        if prefix_all_zeros and future_all_zeros:
            continue

        if np.any([prefix[-1, 0], prefix[-1, 1]]):
            # Prefix trails
            prefix_start_idx = first_nonzero(np.sum(prefix, axis=1), axis=0, invalid_val=-1)
            if not_added_prefix:
                ax.plot(prefix[prefix_start_idx:, 0], prefix[prefix_start_idx:, 1], 'k--', label='History')
                not_added_prefix = False
            else:
                ax.plot(prefix[prefix_start_idx:, 0], prefix[prefix_start_idx:, 1], 'k--')

            # Predicted trails
            if only_predict is None or (only_predict is not None and node_name == only_predict):
                if not_added_samples:
    #                 plt.plot([] , [], 'r', label='Sampled Futures')
                    not_added_samples = False

                for sample_num in range(output_pos[node_name].shape[0]):
                    z_value = tuple(z_values[sample_num])
                    if z_value not in color_dict[node_name]:
                        color_dict[node_name][z_value] = "#%06x" % random.randint(0, 0xFFFFFF)

                    ax.plot(predictions[sample_num, :, 0], predictions[sample_num, :, 1],
                            color=color_dict[node_name][z_value],
                            linewidth=line_width, alpha=line_alpha)

            # Future trails
            future_start_idx = first_nonzero(np.sum(future, axis=1), axis=0, invalid_val=-1)
            future_end_idx = future.shape[0] - first_nonzero(np.sum(future, axis=1)[::-1], axis=0, invalid_val=-1)
            if not_added_future:
                ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                        path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()], label='Actual Future')
                not_added_future = False
            else:
                ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((prefix[-1, 0], prefix[-1, 1]), node_circle_size,
                                facecolor='b' if 'Home' in node_name.type else 'g',
                                edgecolor='k', lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    #         if focus_on:
    #             ax.set_title(node_name)
    #         else:
    #             ax.text(prefix[-1, 0] + 0.4, prefix[-1, 1], node_name, zorder=4)

            if not omit_names:
                ax.text(prefix[-1, 0] + 0.4, prefix[-1, 1], node_name, zorder=4)

    # Robot Node
    if focus_on is None and robot_node is not None:
        prefix_earliest_idx = max(0, t_predict - predict_horizon)
        robot_prefix = inputs[robot_node][0, prefix_earliest_idx : t_predict + 1, 0:2].cpu().numpy()
        robot_future = inputs[robot_node][0, t_predict + 1 : min(t_predict + predict_horizon + 1, traj_length), 0:2].cpu().numpy()

        prefix_all_zeros = not np.any(robot_prefix)
        future_all_zeros = not np.any(robot_future)
        if not (prefix_all_zeros and future_all_zeros):
            ax.plot(robot_prefix[:, 0], robot_prefix[:, 1], 'k--')
            ax.plot(robot_future[:, 0], robot_future[:, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            circle = plt.Circle((robot_prefix[-1, 0], robot_prefix[-1, 1]), node_circle_size,
                                facecolor='b' if 'Home' in robot_node.type else 'g',
                                edgecolor='k', lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            # Radius of influence
            if robot_circle:
                circle = plt.Circle((robot_prefix[-1, 0], robot_prefix[-1, 1]), test_stg.hyperparams['edge_radius'],
                                    fill=False, color='r', linestyle='--', zorder=3)
                ax.plot([], [], 'r--', label='Edge Radius')
                ax.add_artist(circle)

            if not omit_names:
                ax.text(robot_prefix[-1, 0] + 0.4, robot_prefix[-1, 1], robot_node, zorder=4)

    if focus_on is not None:
        y_radius = focus_window_height
        x_radius = aspect_ratio*y_radius
        ax.set_ylim((prefix[-1, 1] - y_radius, prefix[-1, 1] + y_radius))
        ax.set_xlim((prefix[-1, 0] - x_radius, prefix[-1, 0] + x_radius))

    if add_legend:
        ax.legend(loc='best')

    if title is not None:
        ax.set_title(title)

    string_splitter = ' '
    if omit_names:
        string_splitter = '\n'

    if axes_labels:
        ax.set_xlabel('Longitudinal Court Position ($l$)')
        ax.set_ylabel('Lateral Court%sPosition ($w$)' % string_splitter)

    if rotate_axes_text != 0:
        plt.xticks(rotation=rotate_axes_text)
        plt.yticks(rotation=rotate_axes_text)

    fig.tight_layout()

    if return_fig:
        if return_preds:
            return fig, output_pos
        else:
            return fig

    if return_frame:
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png", transparent=True, dpi=dpi)
        buffer_.seek(0)
        data = np.asarray(Image.open( buffer_ ))

        plt.close(fig);
        return data

    if save_at is not None:
        plt.savefig(save_at, dpi=300, transparent=True)

    plt.show()
    plt.close(fig);


def plot_predictions(eval_data_dict, model_registrar,
                     robot_node, hyperparams, device, dt,
                     max_speed, color_dict=None,
                     num_samples=100, data_id=2, t_predict=56,
                     radius_of_influence=3.0,
                     node_circle_size=0.3, focus_on=None,
                     focus_window_height=6, line_alpha=0.7,
                     line_width=0.2, edge_width=2, circle_edge_width=0.5,
                     only_predict=None,
                     dpi=300, fig_height=4,
                     ylim=(0, 50), xlim=(0, 100),
                     figsize=None, tick_fontsize=13,
                     xlabel='$x$ position (m)', ylabel='$y$ position (m)',
                     custom_xticks=None, custom_yticks=None,
                     legend_loc=None,
                     return_frame=False, return_fig=False,
                     printing=False, robot_circle=True,
                     robot_shift_future=0.0, robot_shift='x',
                     add_legend=True, title=None,
                     flip_axes=False, omit_names=False,
                     axes_labels=True, rotate_axes_text=0,
                     save_at=None):

    from model.dyn_stg import SpatioTemporalGraphCVAEModel
    from utils.scene_utils import create_batch_scene_graph

    if color_dict is None:
        # This changes colors per run.
        color_dict = defaultdict(dict)

    if dt is None:
        raise ValueError('You must supply a time delta, the argument dt cannot be None!')

    if figsize is None:
        aspect_ratio = float(xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        figsize=(fig_height*aspect_ratio, fig_height)

    max_time = eval_data_dict['input_dict'][robot_node][data_id, :].shape[0]
    predict_horizon = hyperparams['prediction_horizon']

    if t_predict < hyperparams['minimum_history_length']:
        raise ValueError('ERROR: t_predict must be >= %d' % hyperparams['minimum_history_length'])
    elif t_predict + predict_horizon >= max_time:
        raise ValueError('ERROR: t_predict must be <= %d' % (max_time - predict_horizon - 1))

    ###################
    ### Predictions ###
    ###################
    tic = timeit.default_timer()

    tic0 = timeit.default_timer()

    hyperparams['state_dim'] = eval_data_dict['input_dict'][robot_node].shape[2]
    hyperparams['pred_dim'] = len(eval_data_dict['pred_indices'])
    hyperparams['pred_indices'] = eval_data_dict['pred_indices']
    hyperparams['nodes_standardization'] = eval_data_dict["nodes_standardization"]
    hyperparams["labels_standardization"] = eval_data_dict["labels_standardization"]

    kwargs_dict = {'dynamic_edges': hyperparams['dynamic_edges'],
                   'edge_state_combine_method': hyperparams['edge_state_combine_method'],
                   'edge_influence_combine_method': hyperparams['edge_influence_combine_method']}

    # Create the aggregate scene_graph for all the data, allowing
    # for batching, just like the old one. Then, for speed tests
    # we'll show how much faster this method is than keeping the
    # full version. Can show graphs of forward inference time vs problem size
    # with two lines (using aggregate graph, using online-computed graph).
    input_dict = {k: v[[data_id]] for k, v in eval_data_dict['input_dict'].items()}
    label_dict = {k: v[[data_id]] for k, v in eval_data_dict['labels'].items()}
    robot_future = eval_data_dict['input_dict'][robot_node][[data_id],
                                                            t_predict + 1 : t_predict + predict_horizon + 1]

    if robot_shift_future != 0.0:
        idx = 4 if robot_shift == 'x' else 5
        print('Shifting %s by %.2f!' % (robot_shift, robot_shift_future))
        robot_future[..., idx] += robot_shift_future
        robot_future[..., idx - 2] = cumtrapz(robot_future[..., idx], axis=1, initial=0, dx=dt) + robot_future[0, 0, idx - 2]
        robot_future[..., idx - 4] = cumtrapz(robot_future[..., idx - 2], axis=1, initial=0, dx=dt) + robot_future[0, 0, idx - 4]

    with torch.no_grad():
        test_stg = SpatioTemporalGraphCVAEModel(robot_node, model_registrar, hyperparams, kwargs_dict, None, device)

        test_agg_scene_graph = create_batch_scene_graph(input_dict, float(hyperparams['edge_radius']),
                                                        use_old_method=(hyperparams['dynamic_edges']=='no'))
        if hyperparams['dynamic_edges'] == 'yes':
            test_agg_scene_graph.compute_edge_scaling(hyperparams['edge_addition_filter'], hyperparams['edge_removal_filter'])

        test_stg.set_scene_graph(test_agg_scene_graph)

        test_inputs = {k: torch.from_numpy(v).float() for k, v in input_dict.items() if v.size > 0}
        test_labels = {k: torch.from_numpy(v).float() for k, v in label_dict.items()}

        test_inputs[str(robot_node) + "_future"] = torch.from_numpy(robot_future).float()
        test_inputs['traj_lengths'] = torch.tensor([t_predict])
        if hyperparams['dynamic_edges'] == 'yes':
            test_inputs['edge_scaling_mask'] = torch.from_numpy(test_agg_scene_graph.edge_scaling_mask).float()

        toc0 = timeit.default_timer()
        if printing:
            print("constructing feed_dict took: ", toc0 - tic0, " (s), running pytorch!")

        tic0 = timeit.default_timer()
        outputs = test_stg.predict(test_inputs, hyperparams['prediction_horizon'], num_samples)
        toc0 = timeit.default_timer()

        outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

    if printing:
        print("done running pytorch!, took (s): ", toc0 - tic0)

    toc = timeit.default_timer()

    if printing:
        print("total time taken (s): ", toc - tic)

    ########################
    ### Data Preparation ###
    ########################
    prefixes_dict = dict()
    futures_dict = dict()
    nodes = test_agg_scene_graph.nodes
    prefix_earliest_idx = max(0, t_predict - predict_horizon)
    for node in nodes:
        if robot_node == node:
            continue

        node_data = input_dict[node]
        if printing:
            print('node', node)
            print('node_data.shape', node_data.shape)

        prefixes_dict[node] = node_data[[0], prefix_earliest_idx : t_predict + 1, 0:2]
        futures_dict[node] = node_data[[0], t_predict + 1 : t_predict + predict_horizon + 1]
        if printing:
            print('node', node)
            print('prefixes_dict[node].shape', prefixes_dict[node].shape)
            print('futures_dict[node].shape', futures_dict[node].shape)

    # 44.72 km/h = 40.76 ft/s (ie. that's the max value that a coordinate can be)
    output_pos = dict()
    sampled_zs = dict()
    for node in nodes:
        if robot_node == node:
            continue

        key = str(node) + '/y'
        z_key = str(node) + '/z'

        output_pos[node] = integrate_trajectory(outputs[key], [0, 1],
                                                prefixes_dict[node][0, -1], [0, 1],
                                                dt, output_limit=max_speed,
                                                velocity_in=True)
        sampled_zs[node] = outputs[z_key]

    if printing:
        print('prefixes_dict[node].shape', prefixes_dict[node].shape)
        print('futures_dict[node].shape', futures_dict[node].shape)
        print('output_pos[node].shape', output_pos[node].shape)
        print('sampled_zs[node].shape', sampled_zs[node].shape)

    ######################
    ### Visualizations ###
    ######################
    fig, ax = plt.subplots(figsize=figsize)
    not_added_prefix = True
    not_added_future = True
    not_added_samples = True
    for node_name in prefixes_dict:
        if focus_on is not None and node_name != focus_on:
            continue

        prefix = prefixes_dict[node_name][0]
        future = futures_dict[node_name][0]
        predictions = output_pos[node_name][:, 0]
        z_values = sampled_zs[node_name][:, 0]

        prefix_all_zeros = not np.any(prefix)
        future_all_zeros = not np.any(future)
        if prefix_all_zeros and future_all_zeros:
            continue

        if not (xlim[0] <= prefix[-1, 0] <= xlim[1]) or not (ylim[0] <= prefix[-1, 1] <= ylim[1]):
            continue

        if np.any([prefix[-1, 0], prefix[-1, 1]]):
            # Prefix trails
            prefix_start_idx = first_nonzero(np.sum(prefix, axis=1), axis=0, invalid_val=-1)
            if not_added_prefix:
                ax.plot(prefix[prefix_start_idx:, 0], prefix[prefix_start_idx:, 1], 'k--', label='History')
                not_added_prefix = False
            else:
                ax.plot(prefix[prefix_start_idx:, 0], prefix[prefix_start_idx:, 1], 'k--')

            # Predicted trails
            if only_predict is None or (only_predict is not None and node_name == only_predict):
                if not_added_samples:
                    # plt.plot([] , [], 'r', label='Sampled Futures')
                    not_added_samples = False

                for sample_num in range(output_pos[node_name].shape[0]):
                    z_value = tuple(z_values[sample_num])
                    if z_value not in color_dict[node_name]:
                        color_dict[node_name][z_value] = "#%06x" % random.randint(0, 0xFFFFFF)

                    ax.plot(predictions[sample_num, :, 0], predictions[sample_num, :, 1],
                            color=color_dict[node_name][z_value],
                            linewidth=line_width, alpha=line_alpha)

            # Future trails
            future_start_idx = first_nonzero(np.sum(future, axis=1), axis=0, invalid_val=-1)
            future_end_idx = future.shape[0] - first_nonzero(np.sum(future, axis=1)[::-1], axis=0, invalid_val=-1)
            if not_added_future:
                ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                        path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()], label='Actual Future')
                not_added_future = False
            else:
                ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((prefix[-1, 0], prefix[-1, 1]), node_circle_size,
                                facecolor='b' if 'Home' in node_name.type else 'g',
                                edgecolor='k', lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    #         if focus_on:
    #             ax.set_title(node_name)
    #         else:
    #             ax.text(prefix[-1, 0] + 0.4, prefix[-1, 1], node_name, zorder=4)

            if not omit_names:
                ax.text(prefix[-1, 0] + 0.4, prefix[-1, 1], node_name, zorder=4)

    # Robot Node
    if focus_on is None:
        prefix_earliest_idx = max(0, t_predict - predict_horizon)
        robot_prefix = eval_data_dict['input_dict'][robot_node][data_id, prefix_earliest_idx : t_predict + 1, 0:2]
        prefix_start_idx = first_nonzero(np.sum(robot_prefix, axis=1), axis=0, invalid_val=-1)
#         robot_future = eval_data_dict[robot_node][data_id, t_predict + 1 : t_predict + predict_horizon + 1, 0:2]
        robot_future = robot_future[0, :, 0:2].copy()

        prefix_all_zeros = not np.any(robot_prefix)
        future_all_zeros = not np.any(robot_future)
        if not (prefix_all_zeros and future_all_zeros) and ((xlim[0] <= robot_prefix[-1, 0] <= xlim[1])
                                                            and
                                                            (ylim[0] <= robot_prefix[-1, 1] <= ylim[1])):
            ax.plot(robot_prefix[prefix_start_idx:, 0], robot_prefix[prefix_start_idx:, 1], 'k--')
            ax.plot(robot_future[:, 0], robot_future[:, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            circle = plt.Circle((robot_prefix[-1, 0], robot_prefix[-1, 1]), node_circle_size,
                                facecolor='b' if 'Home' in robot_node.type else 'g',
                                edgecolor='k', lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            # Radius of influence
            if robot_circle:
                circle = plt.Circle((robot_prefix[-1, 0], robot_prefix[-1, 1]), radius_of_influence,
                                    fill=False, color='r', linestyle='--', zorder=3)
                ax.plot([], [], 'r--', label='Edge Radius')
                ax.add_artist(circle)

            if not omit_names:
                ax.text(robot_prefix[-1, 0] + 0.4, robot_prefix[-1, 1], robot_node, zorder=4)

    if focus_on is None:
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
    else:
        y_radius = focus_window_height
        x_radius = aspect_ratio*y_radius
        ax.set_ylim((prefix[-1, 1] - y_radius, prefix[-1, 1] + y_radius))
        ax.set_xlim((prefix[-1, 0] - x_radius, prefix[-1, 0] + x_radius))

    if add_legend:
        if legend_loc is not None:
            leg = ax.legend(loc=legend_loc, handlelength=4)
        else:
            leg = ax.legend(loc='best', handlelength=4)

        for line in leg.get_lines():
            line.set_linewidth(2.25)
        for text in leg.get_texts():
            text.set_fontsize(14)

    if title is not None:
        ax.set_title(title)

    if axes_labels:
        ax.set_xlabel(xlabel, fontsize=tick_fontsize)
        ax.set_ylabel(ylabel, fontsize=tick_fontsize)

    if rotate_axes_text != 0:
        if custom_xticks is not None:
            plt.xticks(custom_xticks, rotation=rotate_axes_text, fontsize=tick_fontsize)
        else:
            plt.xticks(rotation=rotate_axes_text, fontsize=tick_fontsize)

        if custom_yticks is not None:
            plt.yticks(custom_yticks, rotation=rotate_axes_text, fontsize=tick_fontsize)
        else:
            plt.yticks(rotation=rotate_axes_text, fontsize=tick_fontsize)
    else:
        if custom_xticks is not None:
            plt.xticks(custom_xticks, fontsize=tick_fontsize)
        else:
            plt.xticks(fontsize=tick_fontsize)

        if custom_yticks is not None:
            plt.yticks(custom_yticks, fontsize=tick_fontsize)
        else:
            plt.yticks(fontsize=tick_fontsize)

    fig.tight_layout()

    if return_fig:
        return fig

    if return_frame:
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png", transparent=True, dpi=dpi)
        buffer_.seek(0)
        data = np.asarray(Image.open( buffer_ ))

        plt.close(fig);
        return data

    if save_at is not None:
        plt.savefig(save_at, dpi=300, transparent=True)

    plt.show()
    plt.close(fig);


def plot_online_prediction(preds_dict, test_data_dict, data_id,
                           predict_horizon,
                           new_inputs_dict, online_stg,
                           t_predict, robot_future, dt, max_speed,
                           color_dict=None,
                           focus_on=None,
                           focus_window_height=6, line_alpha=0.7,
                           line_width=0.2, edge_width=2, circle_edge_width=0.5,
                           only_predict=None, edge_line_width=1.0,
                           dpi=300, fig_height=4,
                           xlim=(0,100), ylim=(0, 50),
                           figsize=None,
                           return_frame=False, return_fig=False,
                           printing=False, robot_circle=False,
                           add_legend=True, title=None,
                           flip_axes=False, omit_names=False,
                           plot_edges=True, axes_labels=True,
                           rotate_axes_text=0, save_at=None):

    if color_dict is None:
        # This changes colors per run.
        color_dict = defaultdict(dict)

    if dt is None:
        raise ValueError('You must supply a time delta, the argument dt cannot be None!')

    robot_node = online_stg.robot_node

    if figsize is None:
        aspect_ratio = float(xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
        figsize=(fig_height*aspect_ratio, fig_height)

    if t_predict < online_stg.hyperparams['minimum_history_length']:
        return

    ###################
    ### Predictions ###
    ###################
    outputs = {k: v.cpu().numpy() for k, v in preds_dict.items()}

    ########################
    ### Data Preparation ###
    ########################
    output_pos = dict()
    sampled_zs = dict()
    futures_dict = dict()
    for node in online_stg.nodes:
        if robot_node == node:
            continue

        key = str(node) + '/y'
        z_key = str(node) + '/z'

        output_pos[node] = integrate_trajectory(outputs[key], [0, 1],
                                                new_inputs_dict[node][0], [0, 1],
                                                dt, output_limit=max_speed,
                                                velocity_in=True)
        sampled_zs[node] = outputs[z_key]

        node_data = test_data_dict['input_dict'][node]
        futures_dict[node] = node_data[[data_id], t_predict + 1 : t_predict + predict_horizon + 1]

    ######################
    ### Visualizations ###
    ######################
    fig, ax = plt.subplots(figsize=figsize)
    not_added_samples = True
    not_added_future = True
    for node_name in online_stg.nodes:
        if focus_on is not None and node_name != focus_on:
            continue

        predictions = output_pos[node_name][:, 0]
        z_values = sampled_zs[node_name][:, 0]

        # Predicted trails
        if only_predict is None or (only_predict is not None and node_name == only_predict):
            if not_added_samples:
#                 plt.plot([] , [], 'r', label='Sampled Futures')
                not_added_samples = False

            for sample_num in range(output_pos[node_name].shape[0]):
                z_value = tuple(z_values[sample_num])
                if z_value not in color_dict[node_name]:
                    color_dict[node_name][z_value] = "#%06x" % random.randint(0, 0xFFFFFF)

                ax.plot(predictions[sample_num, :, 0], predictions[sample_num, :, 1],
                        color=color_dict[node_name][z_value],
                        linewidth=line_width, alpha=line_alpha, zorder=2)

        # Future trails
        future = futures_dict[node_name][0]
        future_start_idx = first_nonzero(np.sum(future, axis=1), axis=0, invalid_val=-1)
        future_end_idx = future.shape[0] - first_nonzero(np.sum(future, axis=1)[::-1], axis=0, invalid_val=-1)
        if not_added_future:
            ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()], label='Actual Future')
            not_added_future = False
        else:
            ax.plot(future[future_start_idx:future_end_idx, 0], future[future_start_idx:future_end_idx, 1], 'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        # Current Node Position
        circle = plt.Circle((new_inputs_dict[node_name][0, 0], new_inputs_dict[node_name][0, 1]), 0.3,
                            facecolor='b' if 'Home' in node_name.type else 'g',
                            edgecolor='k', lw=circle_edge_width,
                            zorder=3)
        ax.add_artist(circle)

#         if focus_on:
#             ax.set_title(node_name)
#         else:
#             ax.text(prefix[-1, 0] + 0.4, prefix[-1, 1], node_name, zorder=4)

        if not omit_names:
            ax.text(new_inputs_dict[node_name][0, 0] + 0.4, new_inputs_dict[node_name][0, 1], node_name, zorder=4)


    # Robot Node
    if focus_on is None and robot_node is not None:
        robot_future = robot_future[:, 0:2]

        future_all_zeros = not np.any(robot_future)
        if not future_all_zeros and robot_node in new_inputs_dict:
            ax.plot(robot_future[:, 0], robot_future[:, 1], 'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            circle = plt.Circle((new_inputs_dict[robot_node][0, 0], new_inputs_dict[robot_node][0, 1]), 0.3,
                                facecolor='b' if 'Home' in robot_node.type else 'g',
                                edgecolor='k', lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            # Radius of influence
            if robot_circle:
                circle = plt.Circle((new_inputs_dict[robot_node][0, 0], new_inputs_dict[robot_node][0, 1]), online_stg.hyperparams['edge_radius'],
                                    fill=False, color='r', linestyle='--', zorder=3)
                ax.plot([], [], 'r--', label='Edge Radius')
                ax.add_artist(circle)

            if not omit_names:
                ax.text(new_inputs_dict[robot_node][0, 0] + 0.4, new_inputs_dict[robot_node][0, 1], robot_node, zorder=4)

    if plot_edges:
        already_seen_pairs = list()
        for node_A, egdes_and_neighbors in online_stg.scene_graph.node_edges_and_neighbors.items():
            for edge_type, neigbors in egdes_and_neighbors.items():
                for node_B in neigbors:
                    if (node_A, node_B) in already_seen_pairs:
                        continue

                    already_seen_pairs.append((node_B, node_A))

                    if robot_node not in [node_A, node_B]:
                        edge_age = min([online_stg.node_models_dict[str(node_A)].get_mask_for_edge_to(node_B).item(),
                                        online_stg.node_models_dict[str(node_B)].get_mask_for_edge_to(node_A).item()])
                    else:
                        edge_age = 1

                    plt.plot([new_inputs_dict[node_A][0, 0], new_inputs_dict[node_B][0, 0]],
                             [new_inputs_dict[node_A][0, 1], new_inputs_dict[node_B][0, 1]],
                             color='k', lw=edge_line_width, dashes=[edge_age, 1 - edge_age], zorder=-1)

    if focus_on is not None:
        y_radius = focus_window_height
        x_radius = aspect_ratio*y_radius
        ax.set_ylim((prefix[-1, 1] - y_radius, prefix[-1, 1] + y_radius))
        ax.set_xlim((prefix[-1, 0] - x_radius, prefix[-1, 0] + x_radius))

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    if add_legend and robot_circle and not future_all_zeros:
        ax.legend(loc='best')

    if title is not None:
        ax.set_title(title)

    if axes_labels:
        ax.set_xlabel('$x$ Position (m)')
        ax.set_ylabel('$y$ Position (m)')

    if rotate_axes_text != 0:
        plt.xticks(rotation=rotate_axes_text)
        plt.yticks(rotation=rotate_axes_text)

    fig.tight_layout()

    if return_fig:
        return fig

    if return_frame:
        buffer_ = StringIO()
        plt.savefig(buffer_, format="png", transparent=True, dpi=dpi)
        buffer_.seek(0)
        data = np.asarray(Image.open( buffer_ ))

        plt.close(fig);
        return data

    if save_at is not None:
        plt.savefig(save_at, dpi=dpi, transparent=True)

    plt.show()
    plt.close(fig);
