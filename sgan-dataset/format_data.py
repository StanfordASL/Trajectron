import numpy as np
import pandas as pd
import pickle
import glob
from collections import defaultdict

import sys
import os
sys.path.append("../code")
from stg_node import STGNode, convert_to_label_node


def extract_mean_and_std(A, tl, off=0, eps=0.01):
    data = np.concatenate([A[i,:l+off,:] for i, l in enumerate(tl)], axis=0)
    mean = np.zeros((A.shape[-1], ))
    std  = np.zeros((A.shape[-1], ))
    for idx in range(A.shape[-1]):
        idx_data = data[..., idx]
        idx_data = idx_data[idx_data != 0.0]
        if len(idx_data) == 0:
            mean[idx], std[idx] = 0.0, 1.0
        else:
            mean[idx], std[idx] = np.mean(idx_data).astype(np.float32), max(np.std(idx_data).astype(np.float32), eps)
    
    return mean, std


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1

for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        data_dict = dict()
        traj_lengths_list = list()
        input_data_dict_list = list()
        data_dict_path = os.path.join('data', '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join('data', desired_source, data_class)):            
            for file in files:
                if file.endswith('.txt'):                    
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)
                    
                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                    
                    data['frame_id'] -= data['frame_id'].min()
                    
                    data['node_type'] = 'Pedestrian'
                    data['node_name'] = data['track_id'].astype(str)
                    data['node_str'] = data['node_type'] + '/' + data['node_name']
                    data.sort_values('frame_id', inplace=True)
        
                    # This is re-assigning node names to not be so unique,
                    # they don't need to be across data IDs.
                    rename_dict = dict()
                    curr_count_dict = defaultdict(int)
                    for node_str in pd.unique(data['node_str']):
                        if node_str in rename_dict:
                            continue
                        else:
                            node_type, node_name = node_str.split('/')
                            rename_dict[node_str] = '%s/%d' % (node_type, curr_count_dict[node_type])
                            curr_count_dict[node_type] += 1

                    print('Num Nodes:', len(rename_dict))
        #             for node_str in pd.unique(data['node_str']):
        #                 print(node_str, '=>', rename_dict[node_str])

                    def convert_to_new_diff(x): 
                        return int(x / frame_diff)*desired_frame_diff

                    old_max_time = data['frame_id'].max()
                    max_time = convert_to_new_diff(old_max_time) + 1
                    print('Old Max Time', old_max_time, 'New Max Time', max_time)
                    final_dx = 0.4
                    for old_node_str in pd.unique(data['node_str']):
                        node_str = rename_dict[old_node_str]
                        node_type, node_name = node_str.split('/')

                        node = STGNode(node_name, node_type)
                        node_df = data[data['node_str'] == old_node_str]
                        assert np.all(np.diff(node_df['frame_id']) == frame_diff)
                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = convert_to_new_diff(node_df['frame_id'].iloc[0])
                        indices = range(new_first_idx, new_first_idx + node_values.shape[0])

                        input_data_dict[node] = np.zeros((1, max_time, state_dim))
                        input_data_dict[node][0, indices, 0:2] = node_values
                        input_data_dict[node][0, indices, 2:4] = np.gradient(input_data_dict[node][0, indices, 0:2], final_dx, axis=0)
                        input_data_dict[node][0, indices, 4:6] = np.gradient(input_data_dict[node][0, indices, 2:4], final_dx, axis=0)

                    # Chopping down to desired_max_time max time.
                    new_bs = int(np.ceil(max_time/float(desired_max_time)))
                    traj_lengths = np.array([desired_max_time]*(new_bs - 1) + [max_time - desired_max_time*(new_bs - 1)], dtype=int)
                    split_idxs = list()
                    for num_split in range(1, new_bs):
                        split_idxs.append(desired_max_time*num_split)

                    for key in input_data_dict:
                        split_data_list = np.split(input_data_dict[key], split_idxs, axis=1)
                        last_split_len = split_data_list[-1].shape[1]
                        assert last_split_len == traj_lengths[-1]
                        if last_split_len < desired_max_time:
                            new_arr = np.zeros((1, desired_max_time, state_dim))
                            new_arr[0, :last_split_len] = split_data_list[-1].copy()
                            split_data_list[-1] = new_arr

                        input_data_dict[key] = np.concatenate(split_data_list, axis=0)

                    traj_lengths_list.append(traj_lengths)
                    input_data_dict_list.append(input_data_dict)
                    print('Done', full_data_path)
                    
        agg_data_dict = dict()
        for idx, input_data_dict in enumerate(input_data_dict_list):
            for key in input_data_dict:
                if key in agg_data_dict:
                    curr_bs = agg_data_dict[key].shape[0]
                else:
                    curr_bs = 0

                expected_bs = np.sum([x.shape[0] for x in traj_lengths_list[:idx+1]])
                # zeros_bs + curr_bs + input_data_dict[key].shape[0] = expected_bs
                # zeros_bs = expected_bs - curr_bs - input_data_dict[key].shape[0]
                zeros_bs = expected_bs - input_data_dict[key].shape[0] - curr_bs            
                if curr_bs == 0:
                    if zeros_bs == 0:
                        agg_data_dict[key] = input_data_dict[key]
                    else:
                        agg_data_dict[key] = np.concatenate([np.zeros((zeros_bs, desired_max_time, state_dim)), input_data_dict[key]], axis=0)
                else:
                    if zeros_bs == 0:
                        agg_data_dict[key] = np.concatenate([agg_data_dict[key], input_data_dict[key]], axis=0)
                    else:
                        agg_data_dict[key] = np.concatenate([agg_data_dict[key], np.zeros((zeros_bs, desired_max_time, state_dim)), input_data_dict[key]], axis=0)
        
        traj_lengths = np.concatenate(traj_lengths_list, axis=0)
        expected_bs = traj_lengths.shape[0]
        for key in agg_data_dict:
            curr_bs = agg_data_dict[key].shape[0]
            if curr_bs < expected_bs:
                zeros_bs = expected_bs - curr_bs
                agg_data_dict[key] = np.concatenate([agg_data_dict[key], np.zeros((zeros_bs, desired_max_time, state_dim))], axis=0)

        # Trimming down even further!
        refined_train_list = list() # List of lists where the internal lists are groups of humans together.
        random_node_value = next(iter(agg_data_dict.values()))    
        for data_id in range(random_node_value.shape[0]):
            curr_train_list = list()
            for node, node_values in agg_data_dict.items():
                # not all zeros = np.any(node_values)
                if np.any(node_values[data_id]):
                    curr_train_list.append(node_values[data_id])

            if 0 < len(curr_train_list):
                refined_train_list.append(curr_train_list)
        
        list_lengths = [len(lst) for lst in refined_train_list]
        unique, counts = np.unique(list_lengths, return_counts=True)
        print('Unique, Counts')
        print(np.asarray((unique, counts)).T)

        num_peds = max(list_lengths)
        num_scenes = len(refined_train_list)
        print('# Unique Pedestrians:', num_peds)
        print('# Scenes:', num_scenes)
        print('# Pedestrians in each Scene:', list_lengths)

        traj_lens = np.zeros((num_scenes, ), dtype=np.int32)
        for i, node_value_list in enumerate(refined_train_list):
            a = np.sum(np.stack(node_value_list, axis=0), axis=2, keepdims=True)
            traj_len_arr = a.shape[1] - (a!=0)[:,::-1].argmax(1)
            traj_lens[i] = np.amax(traj_len_arr)

        agg_data_dict = dict()
        for i in range(num_peds):
            agg_data_dict[STGNode(str(i), 'Pedestrian')] = np.zeros((num_scenes, desired_max_time, state_dim))

        for curr_list_idx in range(num_scenes):
            node_value_list = refined_train_list[curr_list_idx]
            random_indices = np.random.choice(num_peds, len(node_value_list), replace=False)
            for i, item in enumerate(node_value_list):
                agg_data_dict[STGNode(str(random_indices[i]), 'Pedestrian')][curr_list_idx] = item

        print('Node, # Nonzero Scenes, # Zero Scenes')
        for node in agg_data_dict:
            num_nonzero = 0
            for i in range(num_scenes):
                if np.any(agg_data_dict[node][i]):
                    num_nonzero += 1

            print(node, num_nonzero, num_scenes - num_nonzero)

        data_dict['input_dict'] = agg_data_dict
        data_dict['labels'] = {convert_to_label_node(k): v[..., pred_indices] for k, v in data_dict['input_dict'].items()}

        data_dict['nodes_standardization'] = dict()
        data_dict['labels_standardization'] = dict()
        for node, data in data_dict['input_dict'].items():
            mean, std = extract_mean_and_std(data, traj_lens)
            data_dict['nodes_standardization'][node] = {"mean": mean, "std": std}
            data_dict['labels_standardization'][convert_to_label_node(node)] = {"mean": mean[..., pred_indices], "std": std[..., pred_indices]}

        data_dict['input_dict']['extras'] = np.array([], dtype=np.float32).reshape(list(next(iter(data_dict['input_dict'].values())).shape[:-1]) + [0])
        data_dict['input_dict']['traj_lengths'] = traj_lens

        data_dict['pred_indices'] = pred_indices
        data_dict['extras_mean'] = np.array([], dtype=np.float32)
        data_dict['extras_std'] = np.array([], dtype=np.float32)
        data_dict['dt'] = final_dx
        print('Pred Indices:', pred_indices)

        with open(data_dict_path, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
