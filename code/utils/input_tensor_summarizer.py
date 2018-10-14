from __future__ import absolute_import, division, print_function
import tensorflow as tf

import sys
sys.path.append("..")
from stg_node import *
from utils.learning import *


class InputTensorSummarizer(object):
    def __init__(self, features, labels, 
                 mode, hps, 
                 node, robot_node, 
                 scene_graph, 
                 instance_connected_to_robot,
                 edge_state_combine_method,
                 edge_influence_combine_method,
                 dynamic_edges):

        self.hps = hps
        TD = {}    # tensor_dict
        self.robot_traj = robot_traj = features[robot_node]
        self.our_traj = our_traj = features[node]
        self.extras = extras = features["extras"]
        self.traj_lengths = features["traj_lengths"]
        self.features = features

        if isinstance(hps.pred_indices, dict):
            self.pred_indices = hps.pred_indices[node.type]
        else:
            self.pred_indices = hps.pred_indices
        
        self.node = node
        self.robot_node = robot_node
        self.scene_graph = scene_graph
        self.neighbors_via_edge_type = scene_graph.node_edges_and_neighbors[node]
        self.connected_edge_types = self.neighbors_via_edge_type.keys()
        self.instance_connected_to_robot = instance_connected_to_robot
        self.edge_state_combine_method = edge_state_combine_method
        self.edge_influence_combine_method = edge_influence_combine_method
        self.dynamic_edges = dynamic_edges
        
        our_present = str(self.node) + "_present"
        robot_present = str(self.robot_node) + "_present"
        our_future = str(self.node) + "_future"
        robot_future = str(self.robot_node) + "_future"
        
        self.node_type_connects_to_robot = False
        for edge_type in self.connected_edge_types:
            if 'robot' in edge_type:
                self.node_type_connects_to_robot = True
                break
        
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("data_rearranging"):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    mhl, ph = self.hps.minimum_history_length, self.hps.prediction_horizon
                    self.prediction_timesteps = mhl - 1 + tf.mod(tf.random_uniform(self.traj_lengths.shape,
                                                                                   maxval=2**31-1,
                                                                                   dtype=tf.int32),
                                                                 self.traj_lengths-mhl-ph+1)
                    TD[robot_present] = extract_subtensor_per_batch_element(robot_traj, self.prediction_timesteps)     # [bs, state_dim]
                    TD[our_present] = extract_subtensor_per_batch_element(our_traj, self.prediction_timesteps)     # [bs, state_dim]
                    TD["extras_present"] = extract_subtensor_per_batch_element(extras, self.prediction_timesteps) # [bs, extras_dim]
                    TD[robot_future] = tf.stack([extract_subtensor_per_batch_element(robot_traj, self.prediction_timesteps+i+1)
                                                  for i in range(self.hps.prediction_horizon)], axis=1)           # [bs, ph, state_dim]
                    TD[our_future] = tf.stack([extract_subtensor_per_batch_element(labels, self.prediction_timesteps+i+1)
                                                  for i in range(self.hps.prediction_horizon)], axis=1)           # [bs, ph, state_dim]
                elif mode == tf.estimator.ModeKeys.EVAL:
                    TD[robot_present] = self.extract_ragged_subarray(robot_traj)                                  # [nbs, state_dim]
                    TD[our_present] = self.extract_ragged_subarray(our_traj)                                  # [nbs, state_dim]
                    TD["extras_present"] = self.extract_ragged_subarray(extras)                              # [nbs, extras_dim]
                    TD[robot_future] = tf.stack([self.extract_ragged_subarray(robot_traj, i+1)
                                                  for i in range(self.hps.prediction_horizon)], axis=1)      # [nbs, ph, state_dim]
                    TD[our_future] = tf.stack([self.extract_ragged_subarray(labels, i+1)
                                                  for i in range(self.hps.prediction_horizon)], axis=1)      # [nbs, ph, state_dim]
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    TD[robot_present] = self.extract_subarray_ends(robot_traj)                                    # [bs, state_dim]
                    TD[our_present] = self.extract_subarray_ends(our_traj)                                    # [bs, state_dim]
                    TD["extras_present"] = self.extract_subarray_ends(extras)                                # [bs, extras_dim]
                    TD[robot_future] = features[robot_future]                                              # [bs, ph, state_dim]

                our_prediction_present = tf.concat([TD[our_present][:,p:p+1]
                                                     for p in self.pred_indices], axis=1)                     # [bs/nbs, pred_dim]
                TD["joint_present"] = tf.concat([TD[robot_present], our_prediction_present], axis=1)       # [bs/nbs, state_dim+pred_dim]

        # Node History
        TD["history_encoder"] = self.encode_node_history(mode)
        batch_size = tf.shape(TD["history_encoder"])[0]
        
        # Node Edges
        TD["edge_encoders"] = [self.encode_edge(mode, edge_type, self.neighbors_via_edge_type[edge_type]) for edge_type in self.connected_edge_types] # List of [bs/nbs, enc_rnn_dim]
        TD["total_edge_influence"] = self.encode_total_edge_influence(TD["edge_encoders"], batch_size, mode) # [bs/nbs, 4*enc_rnn_dim]

        # Tiling for multiple samples
        if mode == tf.estimator.ModeKeys.PREDICT:
            # This tiling is done because:
            #   a) we must consider the prediction case where there are many candidate robot future actions,
            #   b) the edge and history encoders are all the same regardless of which candidate future robot action we're evaluating.
            TD["joint_present"] = tf.tile(TD["joint_present"], [tf.shape(features[robot_future])[0], 1])
            TD[robot_present] = tf.tile(TD[robot_present], [tf.shape(features[robot_future])[0], 1])
            TD["history_encoder"] = tf.tile(TD["history_encoder"], [tf.shape(features[robot_future])[0], 1])
            TD["total_edge_influence"] = tf.tile(TD["total_edge_influence"], [tf.shape(features[robot_future])[0], 1])

        concat_list = list()
        
        # Every node has an edge-influence encoder (which could just be zero).
        concat_list.append(TD["total_edge_influence"])  # [bs/nbs, 4*enc_rnn_dim]
            
        # Every node has a history encoder.
        concat_list.append(TD["history_encoder"])       # [bs/nbs, enc_rnn_dim]
                
        if self.instance_connected_to_robot:
            TD[self.robot_node.type + "_robot_future_encoder"] = self.encode_node_future(TD[robot_present], 
                                                                                         TD[robot_future],
                                                                                         mode, 
                                                                                         self.robot_node.type + '_robot') 
                                                                                         # [bs/nbs, 4*enc_rnn_dim]
            concat_list.append(TD[self.robot_node.type + "_robot_future_encoder"])   
            
        else:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            concat_list.append(tf.zeros([batch_size, 4*self.hps.enc_rnn_dim_future[0]]))
            
        TD["x"] = tf.concat(concat_list, axis=1) # [bs/nbs, (4 + 1 + 4)*enc_rnn_dim]
            
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            TD[self.node.type + "_future_encoder"] = self.encode_node_future(TD[our_present], 
                                                                             TD[our_future], 
                                                                             mode, 
                                                                             self.node.type) # [bs/nbs, 4*enc_rnn_dim]
        
        self.tensor_dict = TD

        
    def encode_edge(self, mode, edge_type, connected_nodes):
        with tf.variable_scope(edge_type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("edge_encoder"):
                cell = stacked_rnn_cell(self.hps.rnn_cell,
                                        self.hps.rnn_cell_kwargs,
                                        self.hps.enc_rnn_dim_edge,
                                        self.hps.rnn_io_dropout_keep_prob,
                                        mode)
                input_feature_list = [self.features[node] for node in connected_nodes]
                stacked_edge_states = tf.stack(input_feature_list, axis=0)
                if self.dynamic_edges == 'yes':
                    node_idx = self.scene_graph.nodes.index(self.node)
                    connected_node_idxs = [self.scene_graph.nodes.index(node) for node in connected_nodes]
                    edge_mask = tf.stack([self.features["edge_scaling_mask"][:, :, connected_node_idxs[i], node_idx]
                                                for i in range(len(connected_node_idxs))],
                                         axis=2)

                if self.edge_state_combine_method == 'sum':
                    # Used in Structural-RNN.
                    combined_neighbors = tf.reduce_sum(stacked_edge_states, axis=0)
                    if self.dynamic_edges == 'yes':
                        # Should now be (bs, time)
                        edge_mask = tf.minimum(tf.reduce_sum(edge_mask, axis=2, keepdims=True), 1.)
                    
                elif self.edge_state_combine_method == 'max':
                    # Used in NLP, e.g. max over word embeddings.
                    combined_neighbors = tf.reduce_max(stacked_edge_states, axis=0)
                    if self.dynamic_edges == 'yes':
                        # Should now be (bs, time)
                        edge_mask = tf.reduce_max(edge_mask, axis=2, keepdims=True)
                    
                elif self.edge_state_combine_method == 'mean':
                    # Used in NLP, e.g. mean over word embeddings.
                    combined_neighbors = tf.reduce_mean(stacked_edge_states, axis=0)
                    if self.dynamic_edges == 'yes':
                        # Should now be (bs, time)
                        edge_mask = tf.reduce_mean(edge_mask, axis=2, keepdims=True)
                
                joint_history = tf.concat([combined_neighbors, self.our_traj, self.extras], 2, name="joint_history")
                outputs, _ = tf.nn.dynamic_rnn(cell, joint_history, self.traj_lengths, 
                                               dtype=tf.float32, 
                                               time_major=False) # [bs, max_time, enc_rnn_dim]
                
                if mode == tf.estimator.ModeKeys.TRAIN:
                    if self.dynamic_edges == 'yes':
                        return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps) * extract_subtensor_per_batch_element(edge_mask, self.prediction_timesteps)
                    else:
                        return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps)

                elif mode == tf.estimator.ModeKeys.EVAL:
                    if self.dynamic_edges == 'yes':
                        return self.extract_ragged_subarray(outputs) * self.extract_ragged_subarray(edge_mask)
                    else:
                        return self.extract_ragged_subarray(outputs) # [nbs, enc_rnn_dim]

                elif mode == tf.estimator.ModeKeys.PREDICT:
                    if self.dynamic_edges == 'yes':
                        return self.extract_subarray_ends(outputs) * self.extract_subarray_ends(edge_mask)
                    else:
                        return self.extract_subarray_ends(outputs)   # [bs, enc_rnn_dim]

    def encode_total_edge_influence(self, encoded_edges, batch_size, mode):
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("total_edge_influence_encoder"):
                if self.edge_influence_combine_method == 'sum':
                    stacked_encoded_edges = tf.stack(encoded_edges, axis=0)
                    combined_edges = tf.reduce_sum(stacked_encoded_edges, axis=0)
                    
                elif self.edge_influence_combine_method == 'max':
                    stacked_encoded_edges = tf.stack(encoded_edges, axis=0)
                    combined_edges = tf.reduce_max(stacked_encoded_edges, axis=0)
                    
                elif self.edge_influence_combine_method == 'bi-rnn':
                    if len(encoded_edges) == 0:
                        # Four times because we're trying to mimic a bi-directional 
                        # RNN's output (which is c and h from both ends).
                        combined_edges = tf.zeros([batch_size, 4*self.hps.enc_rnn_dim_edge_influence[0]])
                    
                    else:
                        cell = stacked_rnn_cell(self.hps.rnn_cell,
                                            self.hps.rnn_cell_kwargs,
                                            self.hps.enc_rnn_dim_edge_influence,
                                            self.hps.rnn_io_dropout_keep_prob,
                                            mode)

                        # axis=1 because then we get size [batch_size, max_time, depth]
                        encoded_edges = tf.stack(encoded_edges, axis=1)

                        if self.dynamic_edges == 'yes':
                            # # TODO: Implement edge distance here! Closer affects more.
                            # encoded_edges *= distance_mask
                            pass

                        _, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, encoded_edges,
                                                                   dtype=tf.float32, 
                                                                   time_major=False)

                        combined_edges = tf.concat([unpack_RNN_state(state[0]), unpack_RNN_state(state[1])], axis=1) 
                
                return combined_edges
                
    def encode_node_history(self, mode):
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("history_encoder"):
                cell = stacked_rnn_cell(self.hps.rnn_cell,
                                        self.hps.rnn_cell_kwargs,
                                        self.hps.enc_rnn_dim_history,
                                        self.hps.rnn_io_dropout_keep_prob,
                                        mode)
                node_history = tf.concat([self.our_traj, self.extras], 2, name="node_history")
                outputs, _ = tf.nn.dynamic_rnn(cell, node_history, self.traj_lengths, 
                                               dtype=tf.float32, 
                                               time_major=False) # [bs, max_time, enc_rnn_dim]
                
                if mode == tf.estimator.ModeKeys.TRAIN:
                    return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    return self.extract_ragged_subarray(outputs) # [nbs, enc_rnn_dim]
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    return self.extract_subarray_ends(outputs)   # [bs, enc_rnn_dim]

    def encode_node_future(self, node_present, node_future, mode, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("future_encoder"):
                cell = stacked_rnn_cell(self.hps.rnn_cell,
                                        self.hps.rnn_cell_kwargs,
                                        self.hps.enc_rnn_dim_future,
                                        self.hps.rnn_io_dropout_keep_prob,
                                        mode)

                initial_state = project_to_RNN_initial_state(cell, node_present)
                _, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, node_future,
                                                           initial_state_fw=initial_state,
                                                           dtype = tf.float32,
                                                           time_major=False)
                
                return tf.concat([unpack_RNN_state(state[0]), unpack_RNN_state(state[1])], axis=1)

    def extract_ragged_subarray(self, tensor, offset=0):    # defines a new "batch size", call it "nbs"
        mhl, ph = self.hps.minimum_history_length, self.hps.prediction_horizon
        return tf.boolean_mask(tensor[:,mhl-1+offset:,:],                      #tensor.shape[1]
                               tf.sequence_mask(self.traj_lengths-mhl-ph+1, tf.shape(tensor)[1]-mhl+1-offset))

    def extract_subarray_ends(self, tensor, offset=0):      # ON THE CHOPPING BLOCK
        return extract_subtensor_per_batch_element(tensor, self.traj_lengths-1+offset)
