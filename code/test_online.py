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
from model.online_dyn_stg import OnlineSpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from utils.scene_utils import Scene
from utils import plot_utils


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
    'minimum_history_length': 5,    # 0.5 seconds
    'prediction_horizon': 15,       # 1.5 seconds (at least as far as the loss function is concerned)

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
                    type=str, default='sum')
parser.add_argument("--edge_influence_combine_method", help="the method to use for combining edge influences",
                    type=str, default='bi-rnn')
parser.add_argument('--edge_addition_filter', nargs='+', help="what scaling to use for edges as they're created",
                    type=float, default=[0.25, 0.5, 0.75, 1.0]) # We automatically pad left with 0.0
parser.add_argument('--edge_removal_filter', nargs='+', help="what scaling to use for edges as they're removed",
                    type=float, default=[1.0, 0.0]) # We automatically pad right with 0.0

parser.add_argument("--data_dir", help="what dir to look in for data",
                    type=str, default='../ewap-dataset/data')
parser.add_argument("--test_data_dict", help="what file to load for testing data",
                    type=str, default='eval_data_dict.pkl')
parser.add_argument("--trained_model_dir", help="what trained model to use",
                    type=str, default='../ewap-dataset/logs/models_28_Jan_2019_15_35_20')

parser.add_argument('--device', help='what device to perform testing on',
                    type=str, default='cuda:1')

parser.add_argument('--minimum_history_length', help='how many timesteps of data are required before predictions are released',
                    type=int, default=hyperparams['minimum_history_length'])
parser.add_argument('--prediction_horizon', help='how many timesteps to predict ahead',
                    type=int, default=hyperparams['prediction_horizon'])
parser.add_argument('--num_samples', help='how many samples to take during prediction',
                    type=int, default=hyperparams['k_eval'])
parser.add_argument('--plot_online', help='whether to plot predictions online or not',
                    type=str, default='yes')

parser.add_argument('--seed', help='manual seed to use, default is random',
                    type=int, default=123) # TODO: Make this None.
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
    output_save_dir = 'pred_figs/%s_dyn_edges' % args.dynamic_edges
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

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

    # robot_node = stg_node.STGNode('Al Horford', 'HomeC')
    # max_speed = 40.76

    robot_node = stg_node.STGNode('0', 'Pedestrian')
    max_speed = 12.422222

    # Initial memory usage
    print('%.2f MBs of RAM initially used.' % (memInUse()*1000.))

    # Loading weights from the trained model.
    model_registrar = ModelRegistrar(args.trained_model_dir, args.device)
    model_registrar.load_models(1999)

    hyperparams['state_dim'] = test_data_dict['input_dict'][robot_node].shape[2]
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

    data_id = 11
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
                 'mem_MB': [memInUse()*1000.]}
    print("At t=0, have %d nodes, %d edges which uses %.2f MBs of RAM." % (
            perf_dict['nodes'][0], perf_dict['edges'][0], perf_dict['mem_MB'][0])
         )

    for curr_timestep in range(1, test_data_dict['input_dict']['traj_lengths'][data_id] - args.prediction_horizon + 1):
        robot_future = get_robot_future(robot_node, curr_timestep, 
                                        data_id, test_data_dict, 
                                        args.prediction_horizon)

        new_pos_dict, new_inputs_dict = get_inputs_dict(curr_timestep, data_id, 
                                                        test_data_dict)

        start = time.time()
        preds_dict = online_stg.incremental_forward(robot_future, new_pos_dict, new_inputs_dict, 
                                                    args.prediction_horizon, int(args.num_samples/2))
        end = time.time()

        if args.plot_online == 'yes':
            plot_utils.plot_online_prediction(preds_dict, new_inputs_dict, online_stg, 
                                              curr_timestep, robot_future, 
                                              dt=eval_dt, max_speed=max_speed,
                                              ylim=(2, 9), xlim=(-6, 17),
                                              dpi=150, figsize=(2.2*4, 4),
                                              edge_line_width=0.1, line_width=0.3,
                                              omit_names=True,
                                              save_at=os.path.join(output_save_dir, 'online_pred_%d.png' % curr_timestep))

        perf_dict['time'].append(curr_timestep)
        perf_dict['runtime'].append(end-start)
        perf_dict['frequency'].append(1./(end-start))
        perf_dict['nodes'].append(len(online_stg.scene_graph.active_nodes))
        perf_dict['edges'].append(online_stg.scene_graph.num_edges)
        perf_dict['mem_MB'].append(memInUse()*1000.)
        print("t=%d: took %.2f s (= %.2f Hz) and %d nodes, %d edges uses %.2f MBs of RAM." % (
                perf_dict['time'][-1], perf_dict['runtime'][-1], 
                perf_dict['frequency'][-1], perf_dict['nodes'][-1],
                perf_dict['edges'][-1], perf_dict['mem_MB'][-1])
             )

    plot_utils.plot_performance_metrics(perf_dict, output_save_dir)


def get_robot_future(robot_node, timestep, data_id, data_dict, future_length):
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
