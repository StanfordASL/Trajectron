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
from model.dyn_stg import SpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from utils.scene_utils import create_batch_scene_graph
from utils import plot_utils, eval_utils
from tensorboardX import SummaryWriter

hyperparams = {
    ### Training
    ## Batch Sizes
    'batch_size': 64,
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
                    type=float, default=2.0 * 3.28084)
parser.add_argument("--edge_state_combine_method", help="the method to use for combining edges of the same type",
                    type=str, default='sum')
parser.add_argument("--edge_influence_combine_method", help="the method to use for combining edge influences",
                    type=str, default='attention')
parser.add_argument('--edge_addition_filter', nargs='+', help="what scaling to use for edges as they're created",
                    type=float, default=[0.25, 0.5, 1.0]) # We automatically pad left with 0.0
parser.add_argument('--edge_removal_filter', nargs='+', help="what scaling to use for edges as they're removed",
                    type=float, default=[1.0, 0.5, 0.25]) # We automatically pad right with 0.0
parser.add_argument('--incl_robot_node', help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument("--preloaded_data", help="which dataset to use if using one of the paper's original datasets. One of 'sgan-{eth, hotel, univ, zara1, zara2}'. NOTE: This will overwrite the data_dir, train_data_dict, eval_data_dict, and log_dir arguments",
                    type=str, default=None)

parser.add_argument("--data_dir", help="what dir to look in for data",
                    type=str, default='debug')
parser.add_argument("--train_data_dict", help="what file to load for training data",
                    type=str, default='debug_train_data.pkl')
parser.add_argument("--eval_data_dict", help="what file to load for evaluation data",
                    type=str, default='debug_eval_data.pkl')
parser.add_argument("--log_dir", help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str, default='debug/logs')

parser.add_argument('--device', help='what device to perform training on',
                    type=str, default='cuda:1')
parser.add_argument("--eval_device", help="what device to use during evaluation",
                    type=str, default=None)

parser.add_argument("--num_iters", help="number of iterations to train for",
                    type=int, default=2000)
parser.add_argument('--batch_multiplier', help='how many minibatches to run per iteration of training',
                    type=int, default=1)
parser.add_argument('--batch_size', help='training batch size',
                    type=int, default=hyperparams['batch_size'])
parser.add_argument('--eval_batch_size', help='evaluation batch size',
                    type=int, default=10)
parser.add_argument('--k_eval', help='how many samples to take during evaluation',
                    type=int, default=hyperparams['k_eval'])

parser.add_argument('--seed', help='manual seed to use, default is 123',
                    type=int, default=123)
parser.add_argument('--eval_every', help='how often to evaluate during training, never if None',
                    type=int, default=100)
parser.add_argument('--save_every', help='how often to save during training, never if None',
                    type=int, default=100)
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

if args.eval_device is None:
    args.eval_device = args.device

hyperparams['batch_size'] = args.batch_size
hyperparams['k_eval'] = args.k_eval

if args.preloaded_data is not None:
    if args.preloaded_data.startswith('sgan-'):
        data_source = args.preloaded_data.split('-')[1]

        args.data_dir = '../sgan-dataset/data'
        args.train_data_dict = '%s_train.pkl' % data_source
        args.eval_data_dict = '%s_val.pkl' % data_source
        args.log_dir = '../sgan-dataset/logs/%s' % data_source

        # This is the edge radius to use for the pedestrian datasets.
        # The default one is for the NBA dataset.
        args.edge_radius = 1.5
        args.edge_addition_filter = [0.25, 0.5, 0.75, 1.0]
        args.edge_removal_filter = [1.0, 0.0]

        # 44.72 km/h = 12.42 m/s i.e. that's the max value that a velocity coordinate can be.
        max_speed = 12.422222

else:
    max_speed = 100.

print('-----------------------')
print('| TRAINING PARAMETERS |')
print('-----------------------')
print('| batch_size: %d' % args.batch_size)
print('| batch_multiplier: %d' % args.batch_multiplier)
print('| effective batch size: %d (= %d * %d)' % (args.batch_size * args.batch_multiplier, args.batch_size, args.batch_multiplier))
print('| device: %s' % args.device)
print('| eval_device: %s' % args.eval_device)
print('| max_speed: %s' % max_speed)
print('| edge_radius: %s' % args.edge_radius)
print('| EE state_combine_method: %s' % args.edge_state_combine_method)
print('| EIE scheme: %s' % args.edge_influence_combine_method)
print('| dynamic_edges: %s' % args.dynamic_edges)
print('| robot node: %s' % args.incl_robot_node)
print('| edge_addition_filter: %s' % args.edge_addition_filter)
print('| edge_removal_filter: %s' % args.edge_removal_filter)
print('| MHL: %s' % hyperparams['minimum_history_length'])
print('| PH: %s' % hyperparams['prediction_horizon'])
print('-----------------------')

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    # Create the log and model directiory if they're not present.
    model_dir = os.path.join(args.log_dir,
                             'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    log_writer = SummaryWriter(log_dir=model_dir)

    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_data_dict = pickle.load(f, encoding='latin1')
    train_dt = train_data_dict['dt']
    print('Loaded training data from %s, train_dt = %.2f' % (train_data_path, train_dt))

    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_data_dict = pickle.load(f, encoding='latin1')
        eval_dt = eval_data_dict['dt']
        print('Loaded evaluation data from %s, eval_dt = %.2f' % (eval_data_path, eval_dt))

    if args.incl_robot_node:
        robot_node = stg_node.STGNode('0', 'Pedestrian')
    else:
        robot_node = None

    for key in train_data_dict['input_dict'].keys():
        if isinstance(key, stg_node.STGNode):
            random_node = key
            break

    model_registrar = ModelRegistrar(model_dir, args.device)
    hyperparams['state_dim'] = train_data_dict['input_dict'][random_node].shape[2]
    hyperparams['pred_dim'] = len(train_data_dict['pred_indices'])
    hyperparams['pred_indices'] = train_data_dict['pred_indices']
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['nodes_standardization'] = train_data_dict['nodes_standardization']
    hyperparams['labels_standardization'] = train_data_dict['labels_standardization']
    hyperparams['edge_radius'] = args.edge_radius

    if args.eval_every is not None:
        eval_hyperparams = copy.deepcopy(hyperparams)
        eval_hyperparams['nodes_standardization'] = eval_data_dict["nodes_standardization"]
        eval_hyperparams['labels_standardization'] = eval_data_dict["labels_standardization"]

    kwargs_dict = {'dynamic_edges': hyperparams['dynamic_edges'],
                   'edge_state_combine_method': hyperparams['edge_state_combine_method'],
                   'edge_influence_combine_method': hyperparams['edge_influence_combine_method']}

    stg = SpatioTemporalGraphCVAEModel(robot_node, model_registrar,
                                       hyperparams, kwargs_dict,
                                       None, args.device)
    print('Created training STG model.')

    if args.eval_every is not None:
        # It is important that eval_stg uses the same model_registrar as
        # the stg being trained, otherwise you're just repeatedly evaluating
        # randomly-initialized weights!
        eval_stg = SpatioTemporalGraphCVAEModel(robot_node, model_registrar,
                                                eval_hyperparams, kwargs_dict,
                                                None, args.eval_device)
        print('Created evaluation STG model.')

    # Create the aggregate scene_graph for all the data, allowing
    # for batching, just like the old one. Then, for speed tests
    # we'll show how much faster this method is than keeping the
    # full version. Can show graphs of forward inference time vs problem size
    # with two lines (using aggregate graph, using online-computed graph).
    agg_scene_graph = create_batch_scene_graph(train_data_dict['input_dict'],
                                               float(hyperparams['edge_radius']),
                                               use_old_method=(args.dynamic_edges=='no'))
    print('Created aggregate training scene graph.')

    if args.dynamic_edges == 'yes':
        agg_scene_graph.compute_edge_scaling(args.edge_addition_filter, args.edge_removal_filter)
        train_data_dict['input_dict']['edge_scaling_mask'] = agg_scene_graph.edge_scaling_mask
        print('Computed edge scaling for the training scene graph.')

    stg.set_scene_graph(agg_scene_graph)
    stg.set_annealing_params()

    if args.eval_every is not None:
        eval_agg_scene_graph = create_batch_scene_graph(eval_data_dict['input_dict'],
                                                        float(hyperparams['edge_radius']),
                                                        use_old_method=(args.dynamic_edges=='no'))
        print('Created aggregate evaluation scene graph.')

        if args.dynamic_edges == 'yes':
            eval_agg_scene_graph.compute_edge_scaling(args.edge_addition_filter, args.edge_removal_filter)
            eval_data_dict['input_dict']['edge_scaling_mask'] = eval_agg_scene_graph.edge_scaling_mask
            print('Computed edge scaling for the evaluation scene graph.')

        eval_stg.set_scene_graph(eval_agg_scene_graph)
        eval_stg.set_annealing_params()

    # model_registrar.print_model_names()
    optimizer = optim.Adam(model_registrar.parameters(), lr=hyperparams['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams['learning_decay_rate'])

    # Keeping colors consistent throughout training.
    color_dict = defaultdict(dict)

    print_training_header(newline_start=True)
    for curr_iter in range(args.num_iters):
        # Necessary because we flip the weights contained between GPU and CPU sometimes.
        model_registrar.to(args.device)

        # Setting the current iterator value for internal logging.
        stg.set_curr_iter(curr_iter)

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
        log_writer.add_scalar('dynstg/learning_rate',
                              lr_scheduler.get_lr()[0],
                              curr_iter)
        stg.step_annealers()

        # Zeroing gradients for the upcoming iteration.
        optimizer.zero_grad()

        train_losses = list()
        for mb_num in range(args.batch_multiplier):
            # Obtaining the batch's training loss.
            train_inputs, train_labels = sample_inputs_and_labels(train_data_dict, batch_size=hyperparams['batch_size'])

            # Compute the training loss.
            train_loss = stg.train_loss(train_inputs, train_labels, hyperparams['prediction_horizon']) / args.batch_multiplier
            train_losses.append(train_loss.item())

            # Calculating gradients.
            train_loss.backward()

        # Print training information. Also, no newline here. It's added in at a later line.
        iter_train_loss = sum(train_losses)
        print('{:9} | {:10} | '.format(curr_iter, '%.2f' % iter_train_loss),
              end='', flush=True)

        log_writer.add_histogram('dynstg/train_minibatch_losses', np.asarray(train_losses), curr_iter)
        log_writer.add_scalar('dynstg/train_loss', iter_train_loss, curr_iter)

        # Clipping gradients.
        if hyperparams['grad_clip'] is not None:
            nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])

        # # Logging gradient norms.
        # len_prefix = len('model_dict.')
        # for name, param in model_registrar.named_parameters():
        #     if param.grad is None:
        #         # print(name, 'grad is None')
        #         continue

        #     log_writer.add_scalar('gradient_norms/' + name[len_prefix:],
        #                           param.grad.norm(),
        #                           curr_iter)

        # Performing a gradient step.
        optimizer.step()

        # Freeing up memory.
        del train_loss

        if args.eval_every is not None and (curr_iter + 1) % args.eval_every == 0:
            with torch.no_grad():
                # First plotting training predictions.
                pred_fig = plot_utils.plot_predictions_during_training(stg, train_inputs,
                                                                       hyperparams['prediction_horizon'],
                                                                       num_samples=100,
                                                                       dt=train_dt,
                                                                       max_speed=max_speed,
                                                                       color_dict=color_dict,
                                                                       most_likely=True)
                log_writer.add_figure('dynstg/train_prediction', pred_fig, curr_iter)

                train_mse_batch_errors, train_fse_batch_errors = eval_utils.compute_batch_statistics(stg,
                                                                   train_data_dict,
                                                                   hyperparams['minimum_history_length'],
                                                                   hyperparams['prediction_horizon'],
                                                                   num_samples=100,
                                                                   num_runs=100,
                                                                   dt=train_dt,
                                                                   max_speed=max_speed,
                                                                   robot_node=robot_node)
                log_writer.add_histogram('dynstg/train_mse', train_mse_batch_errors, curr_iter)
                log_writer.add_histogram('dynstg/train_fse', train_fse_batch_errors, curr_iter)

                mse_boxplot_fig, fse_boxplot_fig = plot_utils.plot_boxplots_during_training(train_mse_batch_errors, train_fse_batch_errors)
                log_writer.add_figure('dynstg/train_mse_boxplot', mse_boxplot_fig, curr_iter)
                log_writer.add_figure('dynstg/train_fse_boxplot', fse_boxplot_fig, curr_iter)

                log_writer.add_scalars('dynstg/train_sq_error',
                                       {'mean_mse': torch.mean(train_mse_batch_errors),
                                        'mean_fse': torch.mean(train_fse_batch_errors),
                                        'median_mse': torch.median(train_mse_batch_errors),
                                        'median_fse': torch.median(train_fse_batch_errors)},
                                       curr_iter)

                # Then computing evaluation values and predictions.
                model_registrar.to(args.eval_device)
                eval_stg.set_curr_iter(curr_iter)
                eval_inputs, eval_labels = sample_inputs_and_labels(eval_data_dict,
                                                                    device=args.eval_device,
                                                                    batch_size=args.eval_batch_size)

                (eval_loss_q_is, eval_loss_p, eval_loss_exact) = eval_stg.eval_loss(eval_inputs, eval_labels,
                                                                                    hyperparams['prediction_horizon'])
                log_writer.add_scalars('dynstg/eval',
                                       {'nll_q_is': eval_loss_q_is,
                                        'nll_p': eval_loss_p,
                                        'nll_exact': eval_loss_exact},
                                       curr_iter)

                pred_fig = plot_utils.plot_predictions_during_training(eval_stg, eval_inputs,
                                                                       hyperparams['prediction_horizon'],
                                                                       num_samples=100,
                                                                       dt=eval_dt,
                                                                       max_speed=max_speed,
                                                                       color_dict=color_dict,
                                                                       most_likely=True)
                log_writer.add_figure('dynstg/eval_prediction', pred_fig, curr_iter)

                eval_mse_batch_errors, eval_fse_batch_errors = eval_utils.compute_batch_statistics(eval_stg,
                                                                   eval_data_dict,
                                                                   hyperparams['minimum_history_length'],
                                                                   hyperparams['prediction_horizon'],
                                                                   num_samples=100,
                                                                   num_runs=100,
                                                                   dt=eval_dt,
                                                                   max_speed=max_speed,
                                                                   robot_node=robot_node)
                log_writer.add_histogram('dynstg/eval_mse', eval_mse_batch_errors, curr_iter)
                log_writer.add_histogram('dynstg/eval_fse', eval_fse_batch_errors, curr_iter)

                mse_boxplot_fig, fse_boxplot_fig = plot_utils.plot_boxplots_during_training(eval_mse_batch_errors, eval_fse_batch_errors)
                log_writer.add_figure('dynstg/eval_mse_boxplot', mse_boxplot_fig, curr_iter)
                log_writer.add_figure('dynstg/eval_fse_boxplot', fse_boxplot_fig, curr_iter)

                log_writer.add_scalars('dynstg/eval_sq_error',
                                       {'mean_mse': torch.mean(eval_mse_batch_errors),
                                        'mean_fse': torch.mean(eval_fse_batch_errors),
                                        'median_mse': torch.median(eval_mse_batch_errors),
                                        'median_fse': torch.median(eval_fse_batch_errors)},
                                       curr_iter)

                print('{:15} | {:10} | {:14}'.format('%.2f' % eval_loss_q_is.item(), '%.2f' % eval_loss_p.item(), '%.2f' % eval_loss_exact.item()),
                      end='', flush=True)

                # Freeing up memory.
                del eval_loss_q_is
                del eval_loss_p
                del eval_loss_exact

        else:
            print('{:15} | {:10} | {:14}'.format('', '', ''),
                  end='', flush=True)

        # Here's the newline that ends the current training information printing.
        print('')

        if args.save_every is not None and (curr_iter + 1) % args.save_every == 0:
            model_registrar.save_models(curr_iter)
            print_training_header()


def print_training_header(newline_start=False):
    if newline_start:
        print('')

    print('Iteration | Train Loss | Eval NLL Q (IS) | Eval NLL P | Eval NLL Exact')
    print('----------------------------------------------------------------------')


def sample_inputs_and_labels(data_dict, device=args.device, batch_size=None):
    if batch_size is not None:
        batch_sample = np.random.randint(low=0,
                                         high=data_dict['input_dict']['traj_lengths'].shape[0],
                                         size=batch_size)
        inputs = {k: torch.from_numpy(v[batch_sample]).float() for k, v in data_dict['input_dict'].items() if v.size > 0}
        labels = {k: torch.from_numpy(v[batch_sample]).float() for k, v in data_dict['labels'].items()}
    else:
        inputs = {k: torch.from_numpy(v).float() for k, v in data_dict['input_dict'].items() if v.size > 0}
        labels = {k: torch.from_numpy(v).float() for k, v in data_dict['labels'].items()}

    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = {k: v.to(device) for k, v in labels.items()}

    return inputs, labels


def memInUse():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


if __name__ == '__main__':
    main()

