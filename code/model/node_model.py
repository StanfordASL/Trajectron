import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from model.components import GMM2D, DiscreteLatent, all_one_hot_combinations
from model.model_utils import *


class MultimodalGenerativeCVAE(object):
    def __init__(self, 
                 node, 
                 model_registrar, 
                 robot_node, 
                 kwargs_dict,
                 device,
                 scene_graph=None,
                 log_writer=None):
        self.node = node
        self.model_registrar = model_registrar
        self.robot_node = robot_node
        self.log_writer = log_writer
        self.device = device

        self.node_modules = dict()
        self.edge_state_combine_method = kwargs_dict['edge_state_combine_method']
        self.edge_influence_combine_method = kwargs_dict['edge_influence_combine_method']
        self.dynamic_edges = kwargs_dict['dynamic_edges']
        self.hyperparams = kwargs_dict['hyperparams']

        if scene_graph is not None:
            self.create_graphical_model(scene_graph)


    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter


    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)


    def clear_submodules(self):
        self.node_modules.clear()


    def create_graphical_model(self, scene_graph):
        self.clear_submodules()

        self.scene_graph = scene_graph
        self.neighbors_via_edge_type = scene_graph.node_edges_and_neighbors[self.node]

        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node.type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.hyperparams['state_dim'],
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.node.type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.hyperparams['pred_dim'],
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.node.type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.hyperparams['state_dim'],
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.node.type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.hyperparams['state_dim'],
                                                     self.hyperparams['enc_rnn_dim_future']))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule('robot_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.hyperparams['state_dim'],
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the robot future encoder.
        self.add_submodule('robot_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.hyperparams['state_dim'],
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule('robot_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.hyperparams['state_dim'],
                                                     self.hyperparams['enc_rnn_dim_future']))

        #####################
        #   Edge Encoders   #
        #####################
        # print('create_graphical_model', self.node)
        # print('create_graphical_model', self.neighbors_via_edge_type)
        for edge_type in self.neighbors_via_edge_type:
            # NOTE: The edge input combining happens during calls 
            # to forward or incremental_forward, so we don't create 
            # a model for it here.
            self.add_submodule(edge_type + '/edge_encoder',
                               model_if_absent=nn.LSTM(input_size=2*self.hyperparams['state_dim'],
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       batch_first=True))

        ##############################
        #   Edge Influence Encoder   #
        ##############################
        # NOTE: The edge influence encoding happens during calls 
        # to forward or incremental_forward, so we don't create 
        # a model for it here for the max and sum variants.
        if self.edge_influence_combine_method == 'bi-rnn':
            self.add_submodule(self.node.type + '/edge_influence_encoder',
                               model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                       bidirectional=True,
                                                       batch_first=True))

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        #                    Edge Influence Encoder                        Node History Encoder                    Future Conditional Encoder
        x_size = 4*self.hyperparams['enc_rnn_dim_edge_influence'] + self.hyperparams['enc_rnn_dim_history'] + 4*self.hyperparams['enc_rnn_dim_future']
        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node.type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size
        
        self.add_submodule(self.node.type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node.type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4*self.hyperparams['enc_rnn_dim_future'], self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4*self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node.type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        self.add_submodule(self.node.type + '/decoder/lstm_cell',
                           model_if_absent=nn.LSTMCell(self.hyperparams['pred_dim'] + self.hyperparams['state_dim'] + z_size + x_size, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node.type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node.type + '/decoder/initial_c',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node.type + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'], self.hyperparams['GMM_components']))
        self.add_submodule(self.node.type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'], self.hyperparams['GMM_components']*self.hyperparams['pred_dim']))
        self.add_submodule(self.node.type + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'], self.hyperparams['GMM_components']*self.hyperparams['pred_dim']))
        self.add_submodule(self.node.type + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'], self.hyperparams['GMM_components']))

        for name, module in self.node_modules.items():
            module.to(self.device)


    def create_new_scheduler(self, name, annealer, annealer_kws, 
                             creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, torch.tensor(value_annealer(0), device=self.device))

            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': torch.tensor(value_annealer(0), device=self.device)})
            rsetattr(self, name + '_optimizer', dummy_optimizer)
            
            value_scheduler = CustomLR(dummy_optimizer, 
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)


    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
             annealer=sigmoid_anneal, 
             annealer_kws={
                'start': self.hyperparams['kl_weight_start'],
                'finish': self.hyperparams['kl_weight'],
                'center_step': self.hyperparams['kl_crossover'],
                'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams['kl_sigmoid_divisor']
             },
             creation_condition=((np.abs(self.hyperparams['alpha'] - 1.0) < 1e-3)
                                 and (not self.hyperparams['use_iwae'])))

        self.create_new_scheduler(name='dec_sample_model_prob',
             annealer=sigmoid_anneal, 
             annealer_kws={
                'start': self.hyperparams['dec_sample_model_prob_start'],
                'finish': self.hyperparams['dec_sample_model_prob_final'],
                'center_step': self.hyperparams['dec_sample_model_prob_crossover'],
                'steps_lo_to_hi': self.hyperparams['dec_sample_model_prob_crossover'] / self.hyperparams['dec_sample_model_prob_divisor']
             },
             creation_condition=self.hyperparams['sample_model_during_dec'])

        self.create_new_scheduler(name='latent.temp',
             annealer=exp_anneal, 
             annealer_kws={
                'start': self.hyperparams['tau_init'],
                'finish': self.hyperparams['tau_final'],
                'rate': self.hyperparams['tau_decay_rate']
             })

        self.create_new_scheduler(name='latent.z_logit_clip',
             annealer=sigmoid_anneal, 
             annealer_kws={
                'start': self.hyperparams['z_logit_clip_start'],
                'finish': self.hyperparams['z_logit_clip_final'],
                'center_step': self.hyperparams['z_logit_clip_crossover'],
                'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams['z_logit_clip_divisor']
             },
             creation_condition=self.hyperparams['use_z_logit_clipping'])


    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        self.summarize_annealers()


    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(self.node), annealed_var.replace('.', '/')), rgetattr(self, annealed_var), self.curr_iter)


    def obtain_encoded_tensor_dict(self, mode, features, labels=None, prediction_timesteps=None):
        TD = dict()    # tensor_dict
        self.robot_traj = robot_traj = features[self.robot_node]
        self.our_traj = our_traj = features[self.node]
        # print('our_traj', our_traj)
        self.traj_lengths = features["traj_lengths"]
        self.features = features
        
        self.connected_edge_types = self.neighbors_via_edge_type.keys()
        
        our_present = str(self.node) + "_present"
        robot_present = str(self.robot_node) + "_present"
        our_future = str(self.node) + "_future"
        robot_future = str(self.robot_node) + "_future"
        
        self.node_type_connects_to_robot = False
        for edge_type in self.connected_edge_types:
            if self.robot_node.type in edge_type and self.robot_node in self.neighbors_via_edge_type[edge_type]:
                self.node_type_connects_to_robot = True
                break

        if mode == ModeKeys.TRAIN:
            if prediction_timesteps is None:
                mhl, ph = self.hyperparams['minimum_history_length'], self.hyperparams['prediction_horizon']
                self.prediction_timesteps = mhl - 1 + torch.fmod(torch.randint(low=0, 
                                                                               high=2**31-1, 
                                                                               size=self.traj_lengths.shape).to(self.device),
                                                                 self.traj_lengths-mhl-ph+1).long()
            else:
                self.prediction_timesteps = prediction_timesteps

            TD[robot_present] = extract_subtensor_per_batch_element(robot_traj, self.prediction_timesteps)     # [bs, state_dim]
            # print('TD[robot_present]', TD[robot_present])
            TD[our_present] = extract_subtensor_per_batch_element(our_traj, self.prediction_timesteps)     # [bs, state_dim]
            # print('TD[our_present]', TD[our_present])
            TD[robot_future] = torch.stack([extract_subtensor_per_batch_element(robot_traj, self.prediction_timesteps+i+1)
                                    for i in range(self.hyperparams['prediction_horizon'])], dim=1)           # [bs, ph, state_dim]
            # print('TD[robot_future]', TD[robot_future])
            TD[our_future] = torch.stack([extract_subtensor_per_batch_element(labels, self.prediction_timesteps+i+1)
                                    for i in range(self.hyperparams['prediction_horizon'])], dim=1)           # [bs, ph, state_dim]
            # print('TD[our_future]', TD[our_future])

        elif mode == ModeKeys.EVAL:
            TD[robot_present] = self.extract_ragged_subarray(robot_traj)                                  # [nbs, state_dim]
            TD[our_present] = self.extract_ragged_subarray(our_traj)                                  # [nbs, state_dim]
            TD[robot_future] = torch.stack([self.extract_ragged_subarray(robot_traj, i+1)
                                          for i in range(self.hyperparams['prediction_horizon'])], dim=1)      # [nbs, ph, state_dim]
            TD[our_future] = torch.stack([self.extract_ragged_subarray(labels, i+1)
                                          for i in range(self.hyperparams['prediction_horizon'])], dim=1)      # [nbs, ph, state_dim]

        elif mode == ModeKeys.PREDICT:
            TD[robot_present] = self.extract_subarray_ends(robot_traj)                                    # [bs, state_dim]
            TD[our_present] = self.extract_subarray_ends(our_traj)                                    # [bs, state_dim]
            TD[robot_future] = features[robot_future]                                              # [bs, ph, state_dim]

        our_prediction_present = TD[our_present][:,self.hyperparams['pred_indices']]               # [bs/nbs, pred_dim]
        TD["joint_present"] = torch.cat([TD[robot_present], our_prediction_present], dim=1)        # [bs/nbs, state_dim+pred_dim]

        # Node History
        TD["history_encoder"] = self.encode_node_history(mode)
        # print('TD["history_encoder"]', TD["history_encoder"])
        batch_size = TD["history_encoder"].size()[0]
        
        # Node Edges
        # print('obtain_encoded_tensor_dict', self.node)
        # print('obtain_encoded_tensor_dict', self.connected_edge_types)
        TD["edge_encoders"] = [self.encode_edge(mode, edge_type, self.neighbors_via_edge_type[edge_type]) for edge_type in self.connected_edge_types] # List of [bs/nbs, enc_rnn_dim]
        TD["total_edge_influence"] = self.encode_total_edge_influence(TD["edge_encoders"], batch_size, mode) # [bs/nbs, 4*enc_rnn_dim]

        # Tiling for multiple samples
        if mode == ModeKeys.PREDICT:
            # This tiling is done because:
            #   a) we must consider the prediction case where there are many candidate robot future actions,
            #   b) the edge and history encoders are all the same regardless of which candidate future robot action we're evaluating.
            TD["joint_present"] = TD["joint_present"].repeat(features[robot_future].size()[0], 1)
            TD[robot_present] = TD[robot_present].repeat(features[robot_future].size()[0], 1)
            TD["history_encoder"] = TD["history_encoder"].repeat(features[robot_future].size()[0], 1)
            TD["total_edge_influence"] = TD["total_edge_influence"].repeat(features[robot_future].size()[0], 1)
            
            # Changing it here because we're repeating all our tensors by the number of samples.
            batch_size = TD["history_encoder"].size()[0]

        concat_list = list()
        
        # Every node has an edge-influence encoder (which could just be zero).
        concat_list.append(TD["total_edge_influence"])  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        concat_list.append(TD["history_encoder"])       # [bs/nbs, enc_rnn_dim_history]

        if self.node_type_connects_to_robot:
            TD[self.robot_node.type + "_robot_future_encoder"] = self.encode_robot_future(TD[robot_present], 
                                                                                          TD[robot_future],
                                                                                          mode, 
                                                                                          self.robot_node.type + '_robot') 
                                                                                          # [bs/nbs, 4*enc_rnn_dim_future]
            concat_list.append(TD[self.robot_node.type + "_robot_future_encoder"])   
            
        else:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            concat_list.append(torch.zeros([batch_size, 4*self.hyperparams['enc_rnn_dim_future']], device=self.device))

        # print('self.node_type_connects_to_robot', self.node_type_connects_to_robot)
        # print('edges', self.scene_graph.node_edges_and_neighbors[self.node])
        TD["x"] = torch.cat(concat_list, dim=1) # [bs/nbs, 4*enc_rnn_dim + enc_rnn_dim_history + 4*enc_rnn_dim_future]

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            TD[self.node.type + "_future_encoder"] = self.encode_node_future(TD[our_present], 
                                                                             TD[our_future], 
                                                                             mode, 
                                                                             self.node.type) # [bs/nbs, 4*enc_rnn_dim_future]

        return TD


    def encode_node_history(self, mode):
        node_history = self.our_traj
        # node_history = F.dropout(self.our_traj,
        #                          p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                          training=(mode == ModeKeys.TRAIN))
        if mode == ModeKeys.TRAIN:
            outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node.type + '/node_history_encoder'],
                                                          node_history,
                                                          self.prediction_timesteps)
        else:
            outputs, _ = self.node_modules[self.node.type + '/node_history_encoder'](node_history)

        outputs = F.dropout(outputs,
                            p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN)) # [bs, max_time, enc_rnn_dim]

        if mode == ModeKeys.TRAIN:
            # -1 because outputs is changed by run_lstm_on_variable_length_seqs
            return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps-1) # [bs, enc_rnn_dim]
        elif mode == ModeKeys.EVAL:
            return self.extract_ragged_subarray(outputs) # [nbs, enc_rnn_dim]
        elif mode == ModeKeys.PREDICT:
            return self.extract_subarray_ends(outputs)   # [bs, enc_rnn_dim]


    def encode_edge(self, mode, edge_type, connected_nodes):        
        input_feature_list = [self.features[node] for node in connected_nodes]
        stacked_edge_states = torch.stack(input_feature_list, dim=0)

        if self.dynamic_edges == 'yes':
            node_idx = self.scene_graph.nodes.index(self.node)
            connected_node_idxs = [self.scene_graph.nodes.index(node) for node in connected_nodes]
            edge_mask = torch.stack([self.features["edge_scaling_mask"][:, :, connected_node_idxs[i], node_idx]
                                        for i in range(len(connected_node_idxs))],
                                    dim=2)

        if self.edge_state_combine_method == 'sum':
            # Used in Structural-RNN to combine edges as well.
            combined_neighbors = torch.sum(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be (bs, time)
                edge_mask = torch.clamp(torch.sum(edge_mask, dim=2, keepdim=True), max=1.)
            
        elif self.edge_state_combine_method == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            combined_neighbors = torch.max(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be (bs, time)
                edge_mask = torch.clamp(torch.max(edge_mask, dim=2, keepdim=True), max=1.)
            
        elif self.edge_state_combine_method == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            combined_neighbors = torch.mean(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be (bs, time)
                edge_mask = torch.clamp(torch.mean(edge_mask, dim=2, keepdim=True), max=1.)
        
        joint_history = torch.cat([combined_neighbors, self.our_traj], dim=2)
        # joint_history = F.dropout(joint_history,
        #                           p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                           training=(mode == ModeKeys.TRAIN))
        if mode == ModeKeys.TRAIN:
            outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[edge_type + '/edge_encoder'],
                                                          joint_history,
                                                          self.prediction_timesteps)
        else:
            outputs, _ = self.node_modules[edge_type + '/edge_encoder'](joint_history)

        outputs = F.dropout(outputs,
                            p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN)) # [bs, max_time, enc_rnn_dim]
        
        if mode == ModeKeys.TRAIN:
            # -1 for the ones on outputs because outputs is changed by run_lstm_on_variable_length_seqs
            if self.dynamic_edges == 'yes':
                return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps-1) * extract_subtensor_per_batch_element(edge_mask, self.prediction_timesteps)
            else:
                return extract_subtensor_per_batch_element(outputs, self.prediction_timesteps-1)
        elif mode == ModeKeys.EVAL:
            if self.dynamic_edges == 'yes':
                return self.extract_ragged_subarray(outputs) * self.extract_ragged_subarray(edge_mask)
            else:
                return self.extract_ragged_subarray(outputs) # [nbs, enc_rnn_dim]
        elif mode == ModeKeys.PREDICT:
            if self.dynamic_edges == 'yes':
                return self.extract_subarray_ends(outputs) * self.extract_subarray_ends(edge_mask)
            else:
                return self.extract_subarray_ends(outputs)   # [bs, enc_rnn_dim]
    

    def encode_total_edge_influence(self, encoded_edges, batch_size, mode):
        if self.edge_influence_combine_method == 'sum':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)
            
        elif self.edge_influence_combine_method == 'mean':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)
            
        elif self.edge_influence_combine_method == 'max':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)
            
        elif self.edge_influence_combine_method == 'bi-rnn':
            if len(encoded_edges) == 0:
                # Four times because we're trying to mimic a bi-directional 
                # RNN's output (which is c and h from both ends).
                combined_edges = torch.zeros((batch_size, 4*self.hyperparams['enc_rnn_dim_edge_influence']), device=self.device)
            
            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                # encoded_edges = F.dropout(encoded_edges,
                #                   p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                #                   training=(mode == ModeKeys.TRAIN))
                _, state = self.node_modules[self.node.type + '/edge_influence_encoder'](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(combined_edges,
                                  p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                  training=(mode == ModeKeys.TRAIN))

        return combined_edges


    def encode_node_future(self, node_present, node_future, mode, scope):
        initial_h_model = self.node_modules[self.node.type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node.type + '/node_future_encoder/initial_c']
        
        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        # node_future = F.dropout(node_future,
        #                         p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                         training=(mode == ModeKeys.TRAIN))
        _, state = self.node_modules[self.node.type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state


    def encode_robot_future(self, robot_present, robot_future, mode, scope):
        initial_h_model = self.node_modules['robot_future_encoder/initial_h']
        initial_c_model = self.node_modules['robot_future_encoder/initial_c']
        
        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        # robot_future = F.dropout(robot_future,
        #                         p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                         training=(mode == ModeKeys.TRAIN))
        _, state = self.node_modules['robot_future_encoder'](robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state


    # Creates a new batch size "nbs"
    def extract_ragged_subarray(self, tensor, offset=0):
        mhl, ph = self.hyperparams['minimum_history_length'], self.hyperparams['prediction_horizon']
        mask = torch.zeros((tensor.size()[0], tensor.size()[1], 1), dtype=torch.uint8, device=self.device)
        last_timesteps = (self.traj_lengths - ph + offset).long()
        for i, last_timestep in enumerate(last_timesteps):
            mask[i, :last_timestep] = 1

        selection = torch.masked_select(tensor[:, mhl-1+offset:], mask[:, mhl-1+offset:])
        return torch.reshape(selection, (-1, tensor.size()[-1]))


    def extract_subarray_ends(self, tensor, offset=0):
        return extract_subtensor_per_batch_element(tensor, self.traj_lengths.long()-1+offset)


    def q_z_xy(self, x, y, mode):
        xy = torch.cat([x, y], dim=1)
        # print('q_z_xy/xy', xy)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.node.type + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)), 
                          p=1.-self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.node.type + '/hxy_to_z']

        # if self.log_writer is not None:
        #     self.log_writer.add_scalar('%s/%s' % (str(self.node), 'latent/z_logit_clip'), self.latent.z_logit_clip, self.curr_iter)

        # print('q_z_xy/h', h)
        return self.latent.dist_from_h(to_latent(h), mode)


    def p_z_x(self, x, mode):
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.node.type + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1.-self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.node.type + '/hx_to_z']
        # print('p_z_x/h', h)
        return self.latent.dist_from_h(to_latent(h), mode)


    def project_to_GMM_params(self, tensor):
        log_pis = self.node_modules[self.node.type + '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node.type + '/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node.type + '/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules[self.node.type + '/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs


    def p_y_xz(self, x, z_stacked, TD, mode,
               num_predicted_timesteps, num_samples):
        ph = num_predicted_timesteps
        sample_ct = num_samples
        robot_future = str(self.robot_node) + "_future"
        our_future = str(self.node) + "_future"

        k, GMM_c, pred_dim = sample_ct, self.hyperparams['GMM_components'], self.hyperparams['pred_dim']

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(k, 1)], dim=1)

        cell = self.node_modules[self.node.type + '/decoder/lstm_cell']
        initial_h_model = self.node_modules[self.node.type + '/decoder/initial_h']
        initial_c_model = self.node_modules[self.node.type + '/decoder/initial_c']

        initial_state = (initial_h_model(zx), initial_c_model(zx))
        
        log_pis, mus, log_sigmas, corrs = [], [], [], []
        if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            state = initial_state
            if self.hyperparams['sample_model_during_dec'] and mode == ModeKeys.TRAIN:
                input_ = torch.cat([zx, TD['joint_present'].repeat(k, 1)], dim=1)
                for j in range(ph):
                    h_state, c_state = cell(input_, state)
                    log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state) 
                    y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t, self.hyperparams, self.device,
                                self.hyperparams['log_sigma_min'], self.hyperparams['log_sigma_max']).sample()              # [k;bs, pred_dim]
                    
                    # This is where we pick our output y_t or the true output
                    # our_future to pass into the next cell (we do this with
                    # probability self.dec_sample_model_prob and is only done
                    # during training).
                    mask = td.Bernoulli(probs=self.dec_sample_model_prob).sample((y_t.size()[0], 1))

                    # if self.log_writer is not None:
                    #     self.log_writer.add_scalar('%s/%s' % (str(self.node), 'dec_sample_model_prob'), self.dec_sample_model_prob, self.curr_iter)

                    y_t = mask*y_t + (1 - mask)*(TD[our_future][:,j,:].repeat(k, 1))
                    
                    log_pis.append(log_pi_t)
                    mus.append(mu_t)
                    log_sigmas.append(log_sigma_t)
                    corrs.append(corr_t)

                    dec_inputs = torch.cat([TD[robot_future][:,j,:].repeat(k, 1), y_t], dim=1)
                    input_ = torch.cat([zx, dec_inputs], dim=1)
                    state = (h_state, c_state)
                
                log_pis = torch.stack(log_pis, dim=1)
                mus = torch.stack(mus, dim=1)
                log_sigmas = torch.stack(log_sigmas, dim=1)
                corrs = torch.stack(corrs, dim=1)

            else:
                zx_with_time_dim = zx.unsqueeze(dim=1)                           # [k;bs/nbs, 1, N*K + 2*enc_rnn_dim]
                zx_time_tiled = zx_with_time_dim.repeat(1, ph, 1) 
                dec_inputs = torch.cat([
                    TD["joint_present"].unsqueeze(dim=1),
                    torch.cat([TD[robot_future][:,:ph-1,:], TD[our_future][:, :ph-1,:]], dim=2)
                    ], dim=1)
                
                inputs = torch.cat([zx_time_tiled, dec_inputs.repeat(k, 1, 1)], dim=2)
                outputs = list()
                for j in range(ph):
                    h_state, c_state = cell(inputs[:, j, :], state)
                    outputs.append(h_state)
                    state = (h_state, c_state)

                outputs = torch.stack(outputs, dim=1)
                log_pis, mus, log_sigmas, corrs = self.project_to_GMM_params(outputs)

            if self.log_writer is not None:
                self.log_writer.add_histogram('%s/%s' % (str(self.node), 'GMM_log_pis'), log_pis, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node), 'GMM_mus'), mus, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node), 'GMM_log_sigmas'), log_sigmas, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node), 'GMM_corrs'), corrs, self.curr_iter)

        elif mode == ModeKeys.PREDICT:
            input_ = torch.cat([zx, TD["joint_present"].repeat(k, 1)], dim=1)
            state = initial_state

            log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
            for j in range(ph):
                h_state, c_state = cell(input_, state)
                log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)
                
                y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t, self.hyperparams, self.device,
                            self.hyperparams['log_sigma_min'], self.hyperparams['log_sigma_max']).sample()              # [k;bs, pred_dim]
                
                log_pis.append(log_pi_t)
                mus.append(mu_t)
                log_sigmas.append(log_sigma_t)
                corrs.append(corr_t)
                y.append(y_t)

                dec_inputs = torch.cat([TD[robot_future][:,j,:].repeat(k, 1), y_t], dim=1)
                input_ = torch.cat([zx, dec_inputs], dim=1)
                state = (h_state, c_state)

            log_pis = torch.stack(log_pis, dim=1)
            mus = torch.stack(mus, dim=1)
            log_sigmas = torch.stack(log_sigmas, dim=1)
            corrs = torch.stack(corrs, dim=1)
            sampled_future = torch.reshape(torch.stack(y, dim=1), (k, -1, ph, pred_dim))

        y_dist = GMM2D(torch.reshape(log_pis, [k, -1, ph, GMM_c]),
                       torch.reshape(mus, [k, -1, ph, GMM_c*pred_dim]),
                       torch.reshape(log_sigmas, [k, -1, ph, GMM_c*pred_dim]),
                       torch.reshape(corrs, [k, -1, ph, GMM_c]),
                       self.hyperparams, self.device,
                       self.hyperparams['log_sigma_min'], self.hyperparams['log_sigma_max'])

        if mode == ModeKeys.PREDICT:
            return y_dist, sampled_future
        else:
            return y_dist


    def encoder(self, x, y, mode, num_samples=None):
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        # print('encoder/x', x)
        # print('encoder/y', y)
        self.latent.q_dist = self.q_z_xy(x, y, mode)
        self.latent.p_dist = self.p_z_x(x, mode)

        # print('self.latent.q_dist.mean, self.latent.q_dist.variance', self.latent.q_dist.mean, self.latent.q_dist.variance)
        # print('self.latent.p_dist.mean, self.latent.p_dist.variance', self.latent.p_dist.mean, self.latent.p_dist.variance)
        z = self.latent.sample_q(sample_ct, mode)

        # if self.log_writer is not None:
        #     self.log_writer.add_scalar('data/%s/%s' % (str(self.node), 'latent/temp'), self.latent.temp, self.curr_iter)

        if mode == ModeKeys.TRAIN and self.hyperparams['kl_exact']:
            kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(self.node), self.curr_iter)
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node), 'kl'), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj


    def decoder(self, x, y, z, TD, mode,
                num_predicted_timesteps, num_samples):
        y_dist = self.p_y_xz(x, z, TD, mode, num_predicted_timesteps, num_samples)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(y), max=self.hyperparams['log_p_yt_xz_max'])
        if self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node), 'log_p_yt_xz'), log_p_yt_xz, self.curr_iter)
        
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)        
        return log_p_y_xz


    def train_loss(self, inputs, labels, num_predicted_timesteps, prediction_timesteps=None):
        mode = ModeKeys.TRAIN

        TD = self.obtain_encoded_tensor_dict(mode, inputs, labels, prediction_timesteps)

        z, kl = self.encoder(TD["x"], TD[self.node.type + "_future_encoder"], mode)
        # print('z', z)
        # print('kl', kl)
        log_p_y_xz = self.decoder(TD["x"], TD[str(self.node) + "_future"], z, TD, mode,
                                  num_predicted_timesteps,
                                  self.hyperparams['k'])
        # print('log_p_y_xz', log_p_y_xz)

        if np.abs(self.hyperparams['alpha'] - 1.0) < 1e-3 and not self.hyperparams['use_iwae']:
            log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)                       # [nbs]
            # print('log_p_y_xz_mean', log_p_y_xz_mean)
            log_likelihood = torch.mean(log_p_y_xz_mean)
            # print('log_likelihood', log_likelihood)
            ELBO = log_likelihood - self.kl_weight*kl
            # print('ELBO', ELBO)
            loss = -ELBO
            # print('loss', loss)

            if self.log_writer is not None:
                # self.log_writer.add_scalar('%s/%s' % (str(self.node), 'kl_weight'), self.kl_weight, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node), 'log_p_y_xz'), log_p_y_xz_mean, self.curr_iter)

        else:
            log_q_z_xy = self.latent.q_log_prob(z)    # [k, nbs]
            log_p_z_x = self.latent.p_log_prob(z)     # [k, nbs]
            a = self.hyperparams['alpha']
            log_pp_over_q = log_p_y_xz + log_p_z_x - log_q_z_xy
            log_likelihood = (torch.mean(torch.logsumexp(log_pp_over_q*(1.-a), dim=0)) -\
                              torch.log(self.hyperparams['k'])) / (1.-a)
            loss = -log_likelihood

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node), 'log_likelihood'), log_likelihood, self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node), 'loss'), loss, self.curr_iter)
            self.latent.summarize_for_tensorboard(self.log_writer, str(self.node), self.curr_iter)

        return loss


    def eval_loss(self, inputs, labels, num_predicted_timesteps, 
                  compute_naive=True,
                  compute_exact=True):
        mode = ModeKeys.EVAL
        our_future = str(self.node) + "_future"

        TD = self.obtain_encoded_tensor_dict(mode, inputs, labels)

        ### Importance sampled NLL estimate
        z, _ = self.encoder(TD["x"], TD[self.node.type + "_future_encoder"], mode)      # [k_eval, nbs, N*K]
        log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode, 
                                  num_predicted_timesteps, 
                                  self.hyperparams['k_eval'])           # [k_eval, nbs]
        log_q_z_xy = self.latent.q_log_prob(z)                          # [k_eval, nbs]
        log_p_z_x = self.latent.p_log_prob(z)                           # [k_eval, nbs]
        log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x - log_q_z_xy, dim=0)) -\
                         torch.log(torch.tensor(self.hyperparams['k_eval'], dtype=torch.float, device=self.device))
        nll_q_is = -log_likelihood

        ### Naive sampled NLL estimate
        nll_p = torch.tensor(np.nan)
        if compute_naive:
            z = self.latent.sample_p(self.hyperparams['k_eval'], mode)
            log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode,
                                      num_predicted_timesteps,
                                      self.hyperparams['k_eval'])
            log_likelihood_p = torch.mean(torch.logsumexp(log_p_y_xz, dim=0)) -\
                               torch.log(torch.tensor(self.hyperparams['k_eval'], dtype=torch.float, device=self.device))
            nll_p = -log_likelihood_p

        ### Exact NLL
        nll_exact = torch.tensor(np.nan)
        if compute_exact:
            K, N = self.hyperparams['K'], self.hyperparams['N']
            if K**N < 50:
                nbs = TD["x"].size()[0]
                z_raw = torch.from_numpy(all_one_hot_combinations(N, K).astype(np.float32)).to(self.device).repeat(1, nbs)    # [K**N, nbs*N*K]
                z = torch.reshape(z_raw, (K**N, -1, N*K))                                                  # [K**N, nbs, N*K]
                log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode,
                                          num_predicted_timesteps,
                                          K ** N)                                 # [K**N, nbs]
                log_p_z_x = self.latent.p_log_prob(z)                                              # [K**N, nbs]
                exact_log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x, dim=0))
                
                nll_exact = -exact_log_likelihood

        return nll_q_is, nll_p, nll_exact


    def predict(self, inputs, num_predicted_timesteps, num_samples):
        mode = ModeKeys.PREDICT

        TD = self.obtain_encoded_tensor_dict(mode, inputs)

        self.latent.p_dist = self.p_z_x(TD["x"], mode)
        z = self.latent.sample_p(num_samples, mode)
        y_dist, our_sampled_future = self.p_y_xz(TD["x"], z, TD, mode,
                                                 num_predicted_timesteps,
                                                 num_samples)      # y_dist.mean is [k, bs, ph*state_dim]

        predictions_dict = {str(self.node) + "/y": our_sampled_future,
                            str(self.node) + "/z": z}
        return predictions_dict
