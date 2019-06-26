import torch
import numpy as np
from stg_node import STGNode, convert_to_label_node
from utils import plot_utils
import os
import copy
import pickle
from collections import defaultdict, Counter

from model.online_dyn_stg import SpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from utils.scene_utils import create_batch_scene_graph

from attrdict import AttrDict
import sys
sys.path.append('../ref_impls/SocialGAN-PyTorch')

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, get_dset_path


def compute_mse(predicted_trajs, gt_traj):
    first_nz = plot_utils.first_nonzero(np.sum(gt_traj.cpu().numpy(), axis=2)[0, ::-1], axis=0)
    if first_nz < 0:
        return None

    last_gt_idx = gt_traj.shape[1] - first_nz
    error = torch.norm(predicted_trajs[:, :last_gt_idx] - gt_traj[:, :last_gt_idx], dim=2)
    mse = torch.mean(error, dim=1)
    return mse


def compute_fse(predicted_trajs, gt_traj):
    first_nz = plot_utils.first_nonzero(np.sum(gt_traj.cpu().numpy(), axis=2)[0, ::-1], axis=0)
    if first_nz < 0:
        return None

    last_gt_idx = gt_traj.shape[1] - first_nz
    final_error = torch.norm(predicted_trajs[:, last_gt_idx-1] - gt_traj[:, last_gt_idx-1], dim=1)
    return final_error


def compute_preds_dict_only_agg_errors(preds_dict, gt_dict, data_id,
                                       curr_timestep, prediction_horizon,
                                       error_info_dict):
    gt_dict_nodes = gt_dict['input_dict']
    mse_errors_list = list()
    fse_errors_list = list()
    for node in gt_dict_nodes:
        if (str(node) + '/y') in preds_dict:
            gt_traj_numpy = gt_dict_nodes[node][data_id, curr_timestep:curr_timestep+prediction_horizon+1]
            pred_actions = preds_dict[str(node) + '/y'].cpu().numpy()
            pred_trajs = torch.from_numpy(plot_utils.integrate_trajectory(pred_actions, [0, 1],
                                                                          gt_traj_numpy[0], [0, 1],
                                                                          gt_dict['dt'],
                                                                          output_limit=error_info_dict['output_limit'],
                                                                          velocity_in=True).astype(np.float32))

            gt_traj = torch.unsqueeze(torch.from_numpy(gt_traj_numpy[1:].astype(np.float32)), dim=0)
            mse_val = compute_mse(pred_trajs[:, 0], gt_traj[..., 0:2])
            if mse_val is not None:
                mse_errors_list.append(mse_val)

            fse_val = compute_fse(pred_trajs[:, 0], gt_traj[..., 0:2])
            if fse_val is not None:
                fse_errors_list.append(fse_val)

    if len(mse_errors_list + fse_errors_list) == 0:
        return None, None

    mse_errors_combined = torch.cat(mse_errors_list, dim=0)
    fse_errors_combined = torch.cat(fse_errors_list, dim=0)

    return mse_errors_combined, fse_errors_combined


def compute_preds_dict_error(preds_dicts, gt_dict,
                             data_precondition, dataset_name, method,
                             num_runs,
                             random_scene_idxs, data_ids, t_predicts,
                             prediction_horizon, error_info_dict):
    detailed_error_dict = {'data_precondition': list(),
                           'dataset': list(),
                           'method': list(),
                           'run': list(),
                           'node': list(),
                           'sample': list(),
                           'error_type': list(),
                           'error_value': list()}
    batch_error_dict = {'mse': list(), 'fse': list()}

    for run in range(num_runs):
        random_scene_idx = random_scene_idxs[run]
        data_id = data_ids[random_scene_idx]
        curr_timestep = t_predicts[random_scene_idx] - 1
        preds_dict = preds_dicts[run]

        print(run, data_id, curr_timestep)

        gt_dict_nodes = gt_dict['input_dict']
        mse_errors_list = list()
        fse_errors_list = list()
        for node in gt_dict_nodes:
            if node in preds_dict:
                gt_traj_numpy = gt_dict_nodes[node][data_id, curr_timestep:curr_timestep+prediction_horizon+1]
                pred_actions = preds_dict[node].cpu().numpy()
                pred_trajs = torch.from_numpy(plot_utils.integrate_trajectory(pred_actions, [0, 1],
                                                                              gt_traj_numpy[0], [0, 1],
                                                                              gt_dict['dt'],
                                                                              output_limit=error_info_dict['output_limit'],
                                                                              velocity_in=True).astype(np.float32))

                gt_traj = torch.unsqueeze(torch.from_numpy(gt_traj_numpy[1:].astype(np.float32)), dim=0)
                mse_val = compute_mse(pred_trajs[:, 0], gt_traj[..., 0:2])
                if mse_val is not None:
                    mse_errors_list.append(mse_val)

                    detailed_error_dict['data_precondition'].extend([data_precondition]*mse_val.shape[0])
                    detailed_error_dict['dataset'].extend([dataset_name]*mse_val.shape[0])
                    detailed_error_dict['method'].extend([method]*mse_val.shape[0])
                    detailed_error_dict['run'].extend([run]*mse_val.shape[0])
                    detailed_error_dict['node'].extend([str(node)]*mse_val.shape[0])
                    detailed_error_dict['sample'].extend(list(range(mse_val.shape[0])))
                    detailed_error_dict['error_type'].extend(['mse']*mse_val.shape[0])
                    detailed_error_dict['error_value'].extend(mse_val.cpu().numpy().tolist())

                fse_val = compute_fse(pred_trajs[:, 0], gt_traj[..., 0:2])
                if fse_val is not None:
                    fse_errors_list.append(fse_val)

                    detailed_error_dict['data_precondition'].extend([data_precondition]*fse_val.shape[0])
                    detailed_error_dict['dataset'].extend([dataset_name]*fse_val.shape[0])
                    detailed_error_dict['method'].extend([method]*fse_val.shape[0])
                    detailed_error_dict['run'].extend([run]*fse_val.shape[0])
                    detailed_error_dict['node'].extend([str(node)]*fse_val.shape[0])
                    detailed_error_dict['sample'].extend(list(range(fse_val.shape[0])))
                    detailed_error_dict['error_type'].extend(['fse']*fse_val.shape[0])
                    detailed_error_dict['error_value'].extend(fse_val.cpu().numpy().tolist())

        if len(mse_errors_list + fse_errors_list) == 0:
            continue

        batch_error_dict['mse'].append(torch.cat(mse_errors_list, dim=0))
        batch_error_dict['fse'].append(torch.cat(fse_errors_list, dim=0))

    return batch_error_dict, detailed_error_dict


def compute_batch_statistics(stg, data_dict,
                             minimum_history_length, prediction_horizon,
                             num_samples, num_runs, dt, max_speed, robot_node=None):
    inputs = {k: v for k, v in data_dict['input_dict'].items() if k != 'extras'}
    traj_lengths = inputs['traj_lengths']

    # 2x is just to handle if some violate
    # minimum_history_length < (traj_lengths[data_id] - prediction_horizon)
    data_ids = list(np.random.permutation(traj_lengths.shape[0]))

    error_info_dict = {'output_limit': max_speed}
    batch_error_dict = {'mse': list(), 'fse': list()}
    idx = 0
    count = 0
    while count < num_runs:
        data_id = data_ids[idx]

        # Can't do anything if there's just no room to evaluate the model.
        if minimum_history_length >= (traj_lengths[data_id] - prediction_horizon):
            idx = ((idx + 1) % len(data_ids)) # Wrap around and continue
            continue

        t_predict = np.random.randint(low=minimum_history_length,
                                      high=traj_lengths[data_id] - prediction_horizon)

        curr_inputs = {k: torch.from_numpy(v[[data_id]]).float().to(stg.device) for k, v in inputs.items()}
        if robot_node is not None:
            robot_future = curr_inputs[robot_node][[0], t_predict + 1 : t_predict + prediction_horizon + 1]
            curr_inputs[str(robot_node) + "_future"] = robot_future

        curr_inputs['traj_lengths'] = torch.tensor([t_predict])

        with torch.no_grad():
            preds_dict = stg.predict(curr_inputs, prediction_horizon+1, num_samples, most_likely=True)

        mse_errors, fse_errors = compute_preds_dict_only_agg_errors(preds_dict,
                                                          data_dict,
                                                          data_id,
                                                          t_predict,
                                                          prediction_horizon,
                                                          error_info_dict)

        if mse_errors is not None and fse_errors is not None:
            batch_error_dict['mse'].append(mse_errors)
            batch_error_dict['fse'].append(fse_errors)

        count += 1
        idx = ((idx + 1) % len(data_ids)) # Wrap around and continue

    return torch.cat(batch_error_dict['mse'], dim=0), torch.cat(batch_error_dict['fse'], dim=0)


def compute_sgan_errors(preds_dicts, gt_dicts,
                        data_precondition, dataset_name, num_runs):
    detailed_error_dict = {'data_precondition': list(),
                           'dataset': list(),
                           'method': list(),
                           'run': list(),
                           'node': list(),
                           'sample': list(),
                           'error_type': list(),
                           'error_value': list()}
    batch_error_dict = {'mse': list(), 'fse': list()}

    for run in range(num_runs):
        mse_errors_list = list()
        fse_errors_list = list()

        preds_dict = preds_dicts[run]
        gt_dict = gt_dicts[run]
        for node in gt_dict:
            if node in preds_dict:
                mse_val = compute_mse(preds_dict[node], gt_dict[node])
                if mse_val is not None:
                    mse_errors_list.append(mse_val)

                    detailed_error_dict['data_precondition'].extend([data_precondition]*mse_val.shape[0])
                    detailed_error_dict['dataset'].extend([dataset_name]*mse_val.shape[0])
                    detailed_error_dict['method'].extend(['sgan']*mse_val.shape[0])
                    detailed_error_dict['run'].extend([run]*mse_val.shape[0])
                    detailed_error_dict['node'].extend([str(node)]*mse_val.shape[0])
                    detailed_error_dict['sample'].extend(list(range(mse_val.shape[0])))
                    detailed_error_dict['error_type'].extend(['mse']*mse_val.shape[0])
                    detailed_error_dict['error_value'].extend(mse_val.cpu().numpy().tolist())

                fse_val = compute_fse(preds_dict[node], gt_dict[node])
                if fse_val is not None:
                    fse_errors_list.append(fse_val)

                    detailed_error_dict['data_precondition'].extend([data_precondition]*fse_val.shape[0])
                    detailed_error_dict['dataset'].extend([dataset_name]*fse_val.shape[0])
                    detailed_error_dict['method'].extend(['sgan']*fse_val.shape[0])
                    detailed_error_dict['run'].extend([run]*fse_val.shape[0])
                    detailed_error_dict['node'].extend([str(node)]*fse_val.shape[0])
                    detailed_error_dict['sample'].extend(list(range(fse_val.shape[0])))
                    detailed_error_dict['error_type'].extend(['fse']*fse_val.shape[0])
                    detailed_error_dict['error_value'].extend(fse_val.cpu().numpy().tolist())

        if len(mse_errors_list + fse_errors_list) == 0:
            continue

        batch_error_dict['mse'].append(torch.cat(mse_errors_list, dim=0))
        batch_error_dict['fse'].append(torch.cat(fse_errors_list, dim=0))

    return batch_error_dict, detailed_error_dict


def get_sgan_data_format(our_inputs, what_to_check='all'):
    ped_traj_list = list()
    seq_start_end = list()
    ped_traj_rel_list = list()
    gt_pos_list = list()
    data_ids = list()
    t_predicts = list()

    if what_to_check == 'curr':
        prev_tsteps = 1
        future_tsteps = 1
    elif what_to_check == 'all':
        prev_tsteps = 8
        future_tsteps = 12
    elif what_to_check == 'prev':
        prev_tsteps = 8
        future_tsteps = 1

    for data_id in range(our_inputs['traj_lengths'].shape[0]):
        for t_predict in range(8, int(our_inputs['traj_lengths'][data_id]) - 12, 8):
            orig_len_ped_traj_list = len(ped_traj_list)
            for key, value in our_inputs.items():
                if isinstance(key, STGNode):
                    to_check_pos = value[data_id,
                                         t_predict-prev_tsteps:t_predict+future_tsteps,
                                         :2].cpu().numpy()

                    torch_pos = value[data_id, t_predict-8 : t_predict, :2]
                    if np.all(to_check_pos):
                        gt_pos = value[data_id, t_predict : t_predict + 12, :2]
                        rel_traj = torch.zeros_like(torch_pos)
                        rel_traj[1:] = torch_pos[1:] - torch_pos[:-1]

                        ped_traj_list.append(torch_pos)
                        ped_traj_rel_list.append(rel_traj)
                        gt_pos_list.append(gt_pos)

            if orig_len_ped_traj_list < len(ped_traj_list):
                seq_start_end.append(torch.tensor([orig_len_ped_traj_list, len(ped_traj_list)]))
                data_ids.append(data_id)
                t_predicts.append(t_predict)

    obs_traj = torch.stack(ped_traj_list, dim=1)
    pred_traj_gt = torch.stack(gt_pos_list, dim=1)
    obs_traj_rel = torch.stack(ped_traj_rel_list, dim=1)
    seq_start_end = torch.stack(seq_start_end, dim=0)

    return (obs_traj, pred_traj_gt, obs_traj_rel,
            seq_start_end, data_ids, t_predicts)


def get_our_model_dir(dataset_name):
    if dataset_name == 'eth':
        return 'models_03_Mar_2019_00_23_46'
    elif dataset_name == 'hotel':
        return 'models_03_Mar_2019_00_24_04'
    elif dataset_name == 'univ':
        return 'models_03_Mar_2019_13_48_01'
    elif dataset_name == 'zara1':
        return 'models_03_Mar_2019_13_48_21'
    elif dataset_name == 'zara2':
        return 'models_04_Mar_2019_00_49_31'
    else:
        return None


def get_model_hyperparams(args, dataset_name):
    if None not in [args.edge_state_combine_method, args.edge_influence_combine_method]:
        return {'edge_state_combine_method': args.edge_state_combine_method,
                'edge_influence_combine_method': args.edge_influence_combine_method,
                'best_iter': 1299}
    elif dataset_name == 'eth':
        return {'edge_state_combine_method': 'sum',
                'edge_influence_combine_method': 'attention',
                'best_iter': 1299}
    elif dataset_name == 'hotel':
        return {'edge_state_combine_method': 'sum',
                'edge_influence_combine_method': 'attention',
                'best_iter': 999}
    elif dataset_name == 'univ':
        return {'edge_state_combine_method': 'sum',
                'edge_influence_combine_method': 'attention',
                'best_iter': 1099}
    elif dataset_name == 'zara1':
        return {'edge_state_combine_method': 'sum',
                'edge_influence_combine_method': 'attention',
                'best_iter': 499}
    elif dataset_name == 'zara2':
        return {'edge_state_combine_method': 'sum',
                'edge_influence_combine_method': 'attention',
                'best_iter': 499}
    else:
        return None


def list_compare(a, b):
    return Counter(a) == Counter(b)


def sample_inputs_and_labels(data_dict, device, batch_size=None):
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


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    # generator.cuda()
    generator.train()
    return generator


def extract_our_and_sgan_preds(dataset_name, hyperparams, args, data_precondition='all'):
    print('At %s dataset' % dataset_name)

    ### SGAN LOADING ###
    sgan_model_path = os.path.join(args.sgan_models_path, '_'.join([dataset_name, '12', 'model.pt']))

    checkpoint = torch.load(sgan_model_path, map_location='cpu')
    generator = get_generator(checkpoint)
    _args = AttrDict(checkpoint['args'])
    path = get_dset_path(_args.dataset_name, args.sgan_dset_type)
    print('Evaluating', sgan_model_path, 'on', _args.dataset_name, args.sgan_dset_type)

    _, sgan_data_loader = data_loader(_args, path)

    ### OUR METHOD LOADING ###
    data_dir = '../sgan-dataset/data'
    eval_data_dict_name = '%s_test.pkl' % dataset_name
    log_dir = '../sgan-dataset/logs/%s' % dataset_name
    have_our_model = False
    if os.path.isdir(log_dir):
        have_our_model = True

        trained_model_dir = os.path.join(log_dir, get_our_model_dir(dataset_name))
        eval_data_path = os.path.join(data_dir, eval_data_dict_name)
        with open(eval_data_path, 'rb') as f:
            eval_data_dict = pickle.load(f, encoding='latin1')
        eval_dt = eval_data_dict['dt']
        print('Loaded evaluation data from %s, eval_dt = %.2f' % (eval_data_path, eval_dt))

        # Loading weights from the trained model.
        specific_hyperparams = get_model_hyperparams(args, dataset_name)
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
                       'edge_influence_combine_method': hyperparams['edge_influence_combine_method']}


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

        # It is important that eval_stg uses the same model_registrar as
        # the stg being trained, otherwise you're just repeatedly evaluating
        # randomly-initialized weights!
        eval_stg = SpatioTemporalGraphCVAEModel(None, model_registrar,
                                                eval_hyperparams, kwargs_dict,
                                                None, args.eval_device)
        print('Created evaluation STG model.')

        eval_agg_scene_graph = create_batch_scene_graph(eval_data_dict['input_dict'],
                                                        float(hyperparams['edge_radius']),
                                                        use_old_method=(args.dynamic_edges=='no'))
        print('Created aggregate evaluation scene graph.')

        if args.dynamic_edges == 'yes':
            eval_agg_scene_graph.compute_edge_scaling(args.edge_addition_filter, args.edge_removal_filter)
            eval_data_dict['input_dict']['edge_scaling_mask'] = eval_agg_scene_graph.edge_scaling_mask
            print('Computed edge scaling for the evaluation scene graph.')

        eval_stg.set_scene_graph(eval_agg_scene_graph)
        print('Set the aggregate scene graph.')

        eval_stg.set_annealing_params()

    print('About to begin evaluation computation for %s.' % dataset_name)
    with torch.no_grad():
        eval_inputs, _ = sample_inputs_and_labels(eval_data_dict, device=args.eval_device)

        sgan_preds_list = list()
        sgan_gt_list = list()
        our_preds_list = list()
        our_preds_most_likely_list = list()

        (obs_traj, pred_traj_gt, obs_traj_rel,
         seq_start_end, data_ids, t_predicts) = get_sgan_data_format(eval_inputs, what_to_check=data_precondition)

        num_runs = args.num_runs
        print('num_runs, seq_start_end.shape[0]', args.num_runs, seq_start_end.shape[0])
        if args.num_runs > seq_start_end.shape[0]:
            print('num_runs (%d) > seq_start_end.shape[0] (%d), reducing num_runs to match.' % (num_runs, seq_start_end.shape[0]))
            num_runs = seq_start_end.shape[0]

        samples_list = list()
        for _ in range(args.num_samples):
            pred_traj_fake_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(
                pred_traj_fake_rel, obs_traj[-1]
            )

            samples_list.append(pred_traj_fake)

        random_scene_idxs = np.random.choice(seq_start_end.shape[0],
                                             size=(num_runs,),
                                             replace=False).astype(int)

        sgan_history = defaultdict(dict)
        for run in range(num_runs):
            random_scene_idx = random_scene_idxs[run]
            seq_idx_range = seq_start_end[random_scene_idx]

            agent_preds = dict()
            agent_gt = dict()
            for seq_agent in range(seq_idx_range[0], seq_idx_range[1]):
                agent_preds[seq_agent] = torch.stack([x[:, seq_agent] for x in samples_list], dim=0)
                agent_gt[seq_agent] = torch.unsqueeze(pred_traj_gt[:, seq_agent], dim=0)
                sgan_history[run][seq_agent] = obs_traj[:, seq_agent]

            sgan_preds_list.append(agent_preds)
            sgan_gt_list.append(agent_gt)

        print('Done running SGAN')

        if have_our_model:
            sgan_our_agent_map = dict()
            our_sgan_agent_map = dict()
            for run in range(num_runs):
                print('At our run number', run)
                random_scene_idx = random_scene_idxs[run]
                data_id = data_ids[random_scene_idx]
                t_predict = t_predicts[random_scene_idx] - 1

                curr_inputs = {k: v[[data_id]] for k, v in eval_inputs.items()}
                curr_inputs['traj_lengths'] = torch.tensor([t_predict])

                with torch.no_grad():
                    preds_dict_most_likely = eval_stg.predict(curr_inputs, hyperparams['prediction_horizon'], args.num_samples, most_likely=True)
                    preds_dict_full = eval_stg.predict(curr_inputs, hyperparams['prediction_horizon'], args.num_samples, most_likely=False)

                our_preds_most_likely_list.append(preds_dict_most_likely)
                our_preds_list.append(preds_dict_full)

                for node, value in curr_inputs.items():
                    if isinstance(node, STGNode) and np.any(value[0, t_predict]):
                        curr_prev = value[0, t_predict+1-8 : t_predict+1]
                        for seq_agent, sgan_val in sgan_history[run].items():
                            if torch.norm(curr_prev[:, :2] - sgan_val) < 1e-4:
                                sgan_our_agent_map['%d/%d' % (run, seq_agent)] = node
                                our_sgan_agent_map['%d/%s' % (run, str(node))] = '%d/%d' % (run, seq_agent)

            print('Done running Our Method')

        # Pruning values that aren't in either.
        for run in range(num_runs):
            agent_preds = sgan_preds_list[run]
            agent_gt = sgan_gt_list[run]

            new_agent_preds = dict()
            new_agent_gts = dict()
            for agent in agent_preds.keys():
                run_agent_key = '%d/%d' % (run, agent)
                if run_agent_key in sgan_our_agent_map:
                    new_agent_preds[sgan_our_agent_map[run_agent_key]] = agent_preds[agent]
                    new_agent_gts[sgan_our_agent_map[run_agent_key]] = agent_gt[agent]

            sgan_preds_list[run] = new_agent_preds
            sgan_gt_list[run] = new_agent_gts

        for run in range(num_runs):
            agent_preds_ml = our_preds_most_likely_list[run]
            agent_preds_full = our_preds_list[run]

            new_agent_preds = dict()
            new_agent_preds_full = dict()
            for node in [x for x in agent_preds_ml.keys() if x.endswith('/y')]:
                node_key_list = node.split('/')
                node_obj = STGNode(node_key_list[1], node_key_list[0])
                node_obj_key = '%d/%s' % (run, str(node_obj))
                if node_obj_key in our_sgan_agent_map:
                    new_agent_preds[node_obj] = agent_preds_ml[node]
                    new_agent_preds_full[node_obj] = agent_preds_full[node]

            our_preds_most_likely_list[run] = new_agent_preds
            our_preds_list[run] = new_agent_preds_full

        # Guaranteeing the number of agents are the same.
        for run in range(num_runs):
            assert list_compare(our_preds_most_likely_list[run].keys(), sgan_preds_list[run].keys())
            assert list_compare(our_preds_list[run].keys(), sgan_preds_list[run].keys())
            assert list_compare(our_preds_most_likely_list[run].keys(), our_preds_list[run].keys())
            assert list_compare(sgan_preds_list[run].keys(), sgan_gt_list[run].keys())

    return (our_preds_most_likely_list, our_preds_list,
            sgan_preds_list, sgan_gt_list, eval_inputs, eval_data_dict,
            data_ids, t_predicts, random_scene_idxs, num_runs)
