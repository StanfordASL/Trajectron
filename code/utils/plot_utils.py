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
from collections import defaultdict
import matplotlib.patheffects as pe


def plot_performance_metrics(perf_dict, output_save_dir):
    for key, value in perf_dict.items():
        perf_dict[key] = np.asarray(value)

    fig, ax = plt.subplots()
    ax.scatter(perf_dict['nodes'] + perf_dict['edges'], perf_dict['frequency'])
    ax.set_xlabel('Problem Size (Nodes + Edges)')
    ax.set_ylabel('Frequency (Hz)')
    plt.savefig(os.path.join(output_save_dir, 'freq_vs_problem_size.pdf'), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(perf_dict['nodes'] + perf_dict['edges'], perf_dict['mem_MB'])
    ax.set_xlabel('Problem Size (Nodes + Edges)')
    ax.set_ylabel('Memory Usage (MB)')
    plt.savefig(os.path.join(output_save_dir, 'mem_MB_vs_problem_size.pdf'), dpi=300)
    plt.close(fig)


def integrate_trajectory(input_traj, input_dims,
                         ground_truth_traj, ground_truth_traj_dims,
                         dt, output_limit=None, velocity_in=False):
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

    xd_ = cumtrapz(input_traj[..., input_dims[0]], axis=2, initial=0, dx=dt) + ground_truth_traj[ground_truth_traj_dims[0]]
    yd_ = cumtrapz(input_traj[..., input_dims[1]], axis=2, initial=0, dx=dt) + ground_truth_traj[ground_truth_traj_dims[1]]
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
                                     color_dict=None,
                                     data_id=None, t_predict=56,
                                     focus_on=None, node_circle_size=0.3,
                                     focus_window_height=6, line_alpha=0.7, 
                                     line_width=0.2, edge_width=2, circle_edge_width=0.5,
                                     only_predict=None,
                                     dpi=300, fig_height=4,
                                     return_frame=False, return_fig=True,
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
    figsize=(5, 5)
    traj_lengths = inputs['traj_lengths'].cpu().numpy().astype(int)
    predict_horizon = num_predicted_timesteps

    if data_id is None:
        data_id = np.random.choice([idx for idx, traj_len in enumerate(traj_lengths) if t_predict + predict_horizon < traj_len])

    max_time = inputs[robot_node][data_id, :].shape[0]
    traj_length = traj_lengths[data_id]
    
    if t_predict < test_stg.hyperparams['minimum_history_length']:
        raise ValueError('ERROR: t_predict must be >= %d' % test_stg.hyperparams['minimum_history_length'])
    elif t_predict + predict_horizon >= traj_length:
        raise ValueError('ERROR: t_predict must be <= %d' % (traj_length - predict_horizon - 1))
        
    ###################
    ### Predictions ###
    ###################  
    inputs = {k: v[[data_id]] for k, v in inputs.items()}
    robot_future = inputs[robot_node][[0], t_predict + 1 : t_predict + predict_horizon + 1]
    inputs[str(robot_node) + "_future"] = robot_future
    inputs['traj_lengths'] = torch.tensor([t_predict])

    with torch.no_grad():
        tic0 = timeit.default_timer()
        outputs = test_stg.predict(inputs, num_predicted_timesteps, num_samples)
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
                                                futures_dict[node][0, 0], [0, 1],
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
    if focus_on is None:
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
                                                futures_dict[node][0, 0], [0, 1],
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


def plot_online_prediction(preds_dict, new_inputs_dict, online_stg, 
                           t_predict, robot_future, dt, max_speed,
                           color_dict=None, data_id=0,
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

    predict_horizon = robot_future.shape[0]
    
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
            
    ######################
    ### Visualizations ###
    ######################
    fig, ax = plt.subplots(figsize=figsize)
    not_added_samples = True
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
    if focus_on is None:
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

