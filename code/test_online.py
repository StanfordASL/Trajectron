import torch
from torch import nn, optim
import numpy as np
import os
import time
import copy
import psutil
import pickle
import random
import argparse
import stg_node
import pathlib
from collections import defaultdict
from model.online_dyn_stg import OnlineSpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from utils.scene_utils import Scene
from utils import plot_utils, eval_utils


hyperparams = {
    ### Training
    ## Batch Sizes
    'batch_size': 16,
    ## Learning Rate
    'learning_rate': 0.001,
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

parser.add_argument('--full_preds', help="whether to use our Full model or our Most Likely model (z_best, default)",
                    action='store_true')

parser.add_argument("--data_dir", help="what dir to look in for data",
                    type=str, default='../sgan-dataset/data')
parser.add_argument("--test_data_dict", help="what file to load for testing data",
                    type=str, default=None)
parser.add_argument("--trained_model_dir", help="what trained model to use",
                    type=str, default=None)
parser.add_argument("--trained_model_iter", help="what trained model iteration to use",
                    type=int, default=None)

# parser.add_argument("--data_dir", help="what dir to look in for data",
#                     type=str, default='debug')
# parser.add_argument("--test_data_dict", help="what file to load for testing data",
#                     type=str, default='debug_eval_data.pkl')
# parser.add_argument("--trained_model_dir", help="what trained model to use",
#                     type=str, default='debug/logs/models_17_Feb_2019_00_15_04')

parser.add_argument('--device', help='what device to perform testing on',
                    type=str, default='cpu')

parser.add_argument('--minimum_history_length', help='how many timesteps of data are required before predictions are released',
                    type=int, default=hyperparams['minimum_history_length'])
parser.add_argument('--prediction_horizon', help='how many timesteps to predict ahead',
                    type=int, default=hyperparams['prediction_horizon'])
parser.add_argument('--num_samples', help='how many samples to take during prediction',
                    type=int, default=25)
parser.add_argument('--plot_online', help='whether to plot predictions online or not',
                    type=str, default='yes')

parser.add_argument('--seed', help='manual seed to use, default is random',
                    type=int, default=None)
args = parser.parse_args()
if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from 
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    output_save_dir = 'pred_figs/%s_%s_dyn_edges' % ('full' if args.full_preds else 'z_best', args.dynamic_edges)
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

    if args.test_data_dict is None:
        args.test_data_dict = random.choice(['eth', 'hotel', 'univ', 'zara1', 'zara2']) + '_test.pkl'

    with open(os.path.join(args.data_dir, args.test_data_dict), 'rb') as f:
        test_data_dict = pickle.load(f, encoding='latin1')
    
    # Getting the natural time delta for this dataset.
    eval_dt = test_data_dict['dt']

    for node in test_data_dict['nodes_standardization']:
        for key in test_data_dict['nodes_standardization'][node]:
            test_data_dict['nodes_standardization'][node][key] = torch.from_numpy(test_data_dict['nodes_standardization'][node][key]).float().to(args.device)

    for node in test_data_dict['labels_standardization']:
        for key in test_data_dict['labels_standardization'][node]:
            test_data_dict['labels_standardization'][node][key] = torch.from_numpy(test_data_dict['labels_standardization'][node][key]).float().to(args.device)

    # max_speed = 40.76
    # max_speed = 100.
    max_speed = 12.422222
    if args.incl_robot_node:
        # robot_node = stg_node.STGNode('Al Horford', 'HomeC')
        # robot_node = stg_node.STGNode('0', 'Particle')
        robot_node = stg_node.STGNode('0', 'Pedestrian')
    else:
        robot_node = None

    # Initial memory usage
    init_mem_usage = memInUse()*1000.
    print('%.2f MBs of RAM initially used.' % init_mem_usage)

    # Loading weights from the trained model.
    dataset_name = args.test_data_dict.split("_")[0]
    if args.trained_model_dir is None:
        args.trained_model_dir = os.path.join('../sgan-dataset/logs', dataset_name, eval_utils.get_our_model_dir(dataset_name))

    if args.trained_model_iter is None:
        args.trained_model_iter = eval_utils.get_model_hyperparams(args, dataset_name)['best_iter']

    if args.edge_state_combine_method is None: 
        args.edge_state_combine_method = eval_utils.get_model_hyperparams(args, dataset_name)['edge_state_combine_method']

    if args.edge_influence_combine_method is None: 
        args.edge_influence_combine_method = eval_utils.get_model_hyperparams(args, dataset_name)['edge_influence_combine_method']

    model_registrar = ModelRegistrar(args.trained_model_dir, args.device)
    model_registrar.load_models(args.trained_model_iter)
    
    for key in test_data_dict['input_dict'].keys():
        if isinstance(key, stg_node.STGNode):
            random_node = key
            break

    hyperparams['state_dim'] = test_data_dict['input_dict'][random_node].shape[2]
    hyperparams['pred_dim'] = len(test_data_dict['pred_indices'])
    hyperparams['pred_indices'] = test_data_dict['pred_indices']
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['nodes_standardization'] = test_data_dict['nodes_standardization']
    hyperparams['labels_standardization'] = test_data_dict['labels_standardization']
    hyperparams['edge_radius'] = args.edge_radius

    kwargs_dict = {'dynamic_edges': hyperparams['dynamic_edges'],
                   'edge_state_combine_method': hyperparams['edge_state_combine_method'],
                   'edge_influence_combine_method': hyperparams['edge_influence_combine_method'],
                   'edge_addition_filter': args.edge_addition_filter,
                   'edge_removal_filter': args.edge_removal_filter}

    online_stg = OnlineSpatioTemporalGraphCVAEModel(robot_node, model_registrar, 
                                                    hyperparams, kwargs_dict, 
                                                    args.device)

    data_id = random.randint(0, test_data_dict['input_dict'][random_node].shape[0] - 1)

    print('Looking at the %s sequence, data_id %d' % (dataset_name, data_id))

    init_scene_dict = dict()
    for node, traj_data in test_data_dict['input_dict'].items():
        if isinstance(node, stg_node.STGNode):
            init_scene_dict[str(node)] = traj_data[data_id, 0, :2]

    init_scene_graph = Scene(init_scene_dict).get_graph(args.edge_radius)
    online_stg.set_scene_graph(init_scene_graph)

    perf_dict = {'time': [0], 
                 'runtime': [np.nan],
                 'frequency': [np.nan],
                 'nodes': [len(online_stg.scene_graph.active_nodes)], 
                 'edges': [online_stg.scene_graph.num_edges], 
                 'mem_MB': [memInUse()*1000. - init_mem_usage],
                 'mse': [np.nan],
                 'fse': [np.nan]}
    print("At t=0, have %d nodes, %d edges which uses %.2f MBs of RAM." % (
            perf_dict['nodes'][0], perf_dict['edges'][0], perf_dict['mem_MB'][0])
         )

    # Keeps colors constant throughout the visualization.
    color_dict = defaultdict(dict)
    error_info_dict = {'output_limit': max_speed}
    start_idx = 1
    end_idx = test_data_dict['input_dict']['traj_lengths'][data_id] - args.prediction_horizon + 1
    for curr_timestep in range(start_idx, end_idx):
        robot_future = get_robot_future(robot_node, curr_timestep, 
                                        data_id, test_data_dict, 
                                        args.prediction_horizon)

        new_pos_dict, new_inputs_dict = get_inputs_dict(curr_timestep, data_id, 
                                                        test_data_dict)

        start = time.time()
        preds_dict = online_stg.incremental_forward(robot_future, new_pos_dict, new_inputs_dict, 
                                                    args.prediction_horizon, int(args.num_samples),
                                                    most_likely=(not args.full_preds))
        end = time.time()

        mse_errs, fse_errs = eval_utils.compute_preds_dict_only_agg_errors(preds_dict, test_data_dict, data_id, 
                                                                           curr_timestep, args.prediction_horizon,
                                                                           error_info_dict)
        
        if mse_errs is None and fse_errs is None:
            print('No agents in the scene, stopping!')
            break

        if args.plot_online == 'yes':
            plot_utils.plot_online_prediction(preds_dict, test_data_dict, data_id, 
                                              args.prediction_horizon,
                                              new_inputs_dict, online_stg, 
                                              curr_timestep, robot_future, 
                                              dt=eval_dt, max_speed=max_speed,
                                              color_dict=color_dict,
                                              ylim=(0, 20), xlim=(0, 20),
                                              dpi=150, figsize=(4, 4),
                                              edge_line_width=0.1, line_width=0.5,
                                              omit_names=True,
                                              save_at=os.path.join(output_save_dir, 'online_pred_%d.png' % curr_timestep))

        perf_dict['time'].append(curr_timestep)
        perf_dict['runtime'].append(end-start)
        perf_dict['frequency'].append(1./(end-start))
        perf_dict['nodes'].append(len(online_stg.scene_graph.active_nodes))
        perf_dict['edges'].append(online_stg.scene_graph.num_edges)
        perf_dict['mem_MB'].append(memInUse()*1000. - init_mem_usage)
        perf_dict['mse'].append(mse_errs)
        perf_dict['fse'].append(fse_errs)
        print("t=%d: took %.2f s (= %.2f Hz) w/ MSE %.2f and FSE %.2f and %d nodes, %d edges uses %.2f MBs of RAM." % (
                perf_dict['time'][-1], perf_dict['runtime'][-1], 
                perf_dict['frequency'][-1], torch.mean(perf_dict['mse'][-1]), 
                torch.mean(perf_dict['fse'][-1]), perf_dict['nodes'][-1],
                perf_dict['edges'][-1], perf_dict['mem_MB'][-1])
             )

    if curr_timestep != start_idx:
        plot_utils.plot_performance_metrics(perf_dict, output_save_dir, 
                                            hyperparams['minimum_history_length'],
                                            hyperparams['prediction_horizon'])


def get_robot_future(robot_node, timestep, data_id, data_dict, future_length):
    if robot_node is None:
        return None

    return data_dict['input_dict'][robot_node][data_id, timestep : timestep+future_length]


def get_inputs_dict(timestep, data_id, data_dict):
    new_pos_dict = dict()
    new_inputs_dict = dict()
    for node in data_dict['input_dict']:
        if isinstance(node, stg_node.STGNode):
            state_vec = data_dict['input_dict'][node][data_id, [timestep]]
            if state_vec.any():
                new_inputs_dict[node] = state_vec
                new_pos_dict[node] = state_vec[0, :2]

    return new_pos_dict, new_inputs_dict


def memInUse():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB.
    return memoryUse


if __name__ == '__main__':
    main()
