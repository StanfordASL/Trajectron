import torch
import torch.nn as nn
import torch.nn.functional as F
from model.node_model import MultimodalGenerativeCVAE
from model.model_utils import ModeKeys, unpack_RNN_state
from utils.scene_utils import get_edge_id, DirectionalEdge
from stg_node import STGNode, convert_to_label_node, convert_from_label_node
from collections import defaultdict

# This class implements a streamlined version of MultimodalGenerativeCVAE
# which is optimized for streaming data (i.e. repeatedly predicting with
# a batch size of 1, updating the encoder and calling the decoder
# seperately, etc).
class OnlineMultimodalGenerativeCVAE(MultimodalGenerativeCVAE):
    def __init__(self,
                 node,
                 model_registrar,
                 robot_node,
                 kwargs_dict,
                 device,
                 scene_graph=None):
        super(OnlineMultimodalGenerativeCVAE, self).__init__(node,
                 model_registrar,
                 robot_node,
                 kwargs_dict,
                 device,
                 scene_graph,
                 log_writer=None)

        self.edge_addition_filter = torch.tensor(kwargs_dict['edge_addition_filter'], device=device)
        self.edge_removal_filter = torch.tensor(kwargs_dict['edge_removal_filter'], device=device)
        const_filter = [torch.tensor(1.0, device=device)]

        self.curr_hidden_states = dict()

        # This will be a dict of:
        # DirectionalEdge instance : {mask_arr: [x.xx, ...], age: xx}
        self.edge_properties = dict()

        # These maintain information about edges that have been removed.
        # DirectionalEdge.type : [node_A, node_B, ...]
        self.removed_neighbors_by_edge_type = defaultdict(list)
        # {DirectionalEdge instanceA, DirectionalEdge instanceB, ...}
        self.removed_edges = set()

        if scene_graph is not None:
            for edge_type, neighbor_nodes in scene_graph.node_edges_and_neighbors[self.node].items():
                for other_node in neighbor_nodes:
                    # Starting it off with an edge_mask of 1 for edges at the beginning.
                    self.edge_properties[self.get_edge_to(other_node)] = {'mask_arr': const_filter, 'age': 0}


    def get_edge_to(self, other_node):
        return DirectionalEdge(self.node, other_node)


    def get_mask_for_edge_to(self, other_node):
        edge = self.get_edge_to(other_node)
        mask_arr = self.edge_properties[edge]["mask_arr"]
        edge_age = self.edge_properties[edge]["age"]
        return mask_arr[edge_age]


    def update_graph(self, new_scene_graph, new_neighbors, removed_neighbors):
        self.scene_graph = new_scene_graph
        self.neighbors_via_edge_type = new_scene_graph.node_edges_and_neighbors[self.node]

        if self.node in new_neighbors:
            for edge_type, new_neighbor_nodes in new_neighbors[self.node].items():
                self.add_edge_model(edge_type, new_neighbor_nodes)

        if self.node in removed_neighbors:
            for edge_type, removed_neighbor_nodes in removed_neighbors[self.node].items():
                self.remove_edge_model(edge_type, removed_neighbor_nodes)


    def add_edge_model(self, edge_type, new_neighbor_nodes):
        for other_node in new_neighbor_nodes:
            edge = self.get_edge_to(other_node)
            self.edge_properties[edge] = {'mask_arr': self.edge_addition_filter, 'age': 0}

            if edge in self.removed_edges:
                self.removed_neighbors_by_edge_type[edge.type].remove(other_node)
                self.removed_edges.remove(edge)

        if edge_type + '/edge_encoder' not in self.node_modules:
            self.add_submodule(edge_type + '/edge_encoder',
                       model_if_absent=nn.LSTM(input_size=2*self.hyperparams['state_dim'],
                                               hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                               batch_first=True))


    def remove_edge_model(self, edge_type, removed_neighbor_nodes):
        for other_node in removed_neighbor_nodes:
            edge = self.get_edge_to(other_node)
            self.edge_properties[edge] = {'mask_arr': self.edge_removal_filter, 'age': 0}
            self.removed_neighbors_by_edge_type[edge.type].append(other_node)
            self.removed_edges.add(edge)

        if edge_type not in self.neighbors_via_edge_type:
            del self.node_modules[edge_type + '/edge_encoder']


    def encoder_forward(self, new_inputs_dict, robot_future):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        self.TD = self.obtain_encoded_tensor_dict(new_inputs_dict, robot_future)

        # Updating edge ages and removed edges.
        # Need the list(...) wrapper as we're editing the dict while looping over it.
        for edge, edge_props in list(self.edge_properties.items()):
            age = edge_props['age']
            mask_arr = edge_props['mask_arr']
            if (edge in self.removed_edges and age >= (len(mask_arr) - 1)):
                self.removed_neighbors_by_edge_type[edge.type].remove(edge.other_node)
                self.removed_edges.remove(edge)
                del self.edge_properties[edge]

            elif age < len(mask_arr) - 1:
                self.edge_properties[edge]['age'] += 1

        self.latent.p_dist = self.p_z_x(self.TD["x"], mode)


    def obtain_encoded_tensor_dict(self, new_inputs_dict, robot_future):
        TD = dict()    # tensor_dict
        connected_edge_types = self.neighbors_via_edge_type.keys()

        our_present = str(self.node) + "_present"
        if self.robot_node is not None:
            robot_present = str(self.robot_node) + "_present"

        self.node_connects_to_robot = False
        if self.robot_node is not None:
            for edge_type in connected_edge_types:
                if ((self.robot_node.type in edge_type) and
                    (self.robot_node in self.neighbors_via_edge_type[edge_type])):
                    self.node_connects_to_robot = True
                    break

        TD[our_present] = new_inputs_dict[self.node] # [bs, state_dim]
        if self.node_connects_to_robot:
            TD[robot_present + "_orig"] = new_inputs_dict[self.robot_node] # [bs, state_dim]
        elif self.robot_node is not None:
            TD[robot_present + "_orig"] = torch.zeros(1, self.hyperparams['state_dim'])

        our_prediction_present = TD[our_present][:,self.hyperparams['pred_indices']]        # [bs/nbs, pred_dim]
        if self.robot_node is not None:
            TD["joint_present_orig"] = torch.cat([TD[robot_present + "_orig"], our_prediction_present], dim=1) # [bs/nbs, state_dim+pred_dim]
        else:
            TD["joint_present_orig"] = our_prediction_present

        # Node History
        TD["history_encoder_orig"] = self.encode_node_history(TD[our_present])
        # print('TD["history_encoder_orig"]', TD["history_encoder_orig"])
        batch_size = TD["history_encoder_orig"].size()[0]

        # Node Edges
        # print('obtain_encoded_tensor_dict', self.node)
        # print('obtain_encoded_tensor_dict', connected_edge_types)
        for edge_type in self.removed_neighbors_by_edge_type:
            for node in self.removed_neighbors_by_edge_type[edge_type]:
                # This is adding zeros to the inputs of encoders for removed
                # edges (until their influence is completely removed).
                new_inputs_dict[node] = torch.zeros(1, self.hyperparams['state_dim'])

        TD["edge_encoders_orig"] = [self.encode_edge(edge_type, self.neighbors_via_edge_type[edge_type], new_inputs_dict) for edge_type in connected_edge_types] # List of [bs/nbs, enc_rnn_dim]
        TD["total_edge_influence_orig"] = self.encode_total_edge_influence(TD["edge_encoders_orig"], TD["history_encoder_orig"], batch_size) # [bs/nbs, 4*enc_rnn_dim]

        TD["x"] = self.create_encoder_rep(TD, robot_future)
        return TD


    def create_encoder_rep(self, TD, robot_future=None):
        if self.robot_node is not None:
            robot_present = str(self.robot_node) + "_present"

        if self.robot_node is not None and robot_future is not None:
            robot_future_str = str(self.robot_node) + "_future"
            # Updating the robot_future in the TD.
            TD[robot_future_str] = torch.unsqueeze(robot_future, dim=0)

            # Tiling for multiple samples
            # This tiling is done because:
            #   a) we must consider the prediction case where there are many candidate robot future actions,
            #   b) the edge and history encoders are all the same regardless of which candidate future robot action we're evaluating.
            TD["joint_present"] = TD["joint_present_orig"].repeat(TD[robot_future_str].size()[0], 1)
            TD[robot_present] = TD[robot_present + "_orig"].repeat(TD[robot_future_str].size()[0], 1)
            TD["history_encoder"] = TD["history_encoder_orig"].repeat(TD[robot_future_str].size()[0], 1)
            TD["total_edge_influence"] = TD["total_edge_influence_orig"].repeat(TD[robot_future_str].size()[0], 1)

        else:
            TD["joint_present"] = TD["joint_present_orig"]
            TD["history_encoder"] = TD["history_encoder_orig"]
            TD["total_edge_influence"] = TD["total_edge_influence_orig"]
            if self.robot_node is not None:
                TD[robot_present] = TD[robot_present + "_orig"]

        # Changing it here because we're repeating all our tensors by the number of samples.
        batch_size = TD["history_encoder"].size()[0]

        # Holds what will be concatenated to make TD["x"]
        concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        concat_list.append(TD["total_edge_influence"])  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        concat_list.append(TD["history_encoder"])       # [bs/nbs, enc_rnn_dim_history]

        if ((self.robot_node is not None) and
            (self.node_connects_to_robot) and
            (robot_future is not None)):
            TD[self.robot_node.type + "_robot_future_encoder"] = self.encode_robot_future(TD[robot_present],
                                                                                          TD[robot_future_str],
                                                                                          ModeKeys.PREDICT,
                                                                                          self.robot_node.type + '_robot')
                                                                                          # [bs/nbs, 4*enc_rnn_dim_future]
            concat_list.append(TD[self.robot_node.type + "_robot_future_encoder"])

        elif self.robot_node is not None:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            concat_list.append(torch.zeros([batch_size, 4*self.hyperparams['enc_rnn_dim_future']], device=self.device))

        # print('self.node_connects_to_robot', self.node_connects_to_robot)
        # print('edges', self.scene_graph.node_edges_and_neighbors[self.node])
        return torch.cat(concat_list, dim=1) # [bs/nbs, 4*enc_rnn_dim + enc_rnn_dim_history + 4*enc_rnn_dim_future]


    def encode_node_history(self, new_state):
        new_state = torch.unsqueeze(new_state, dim=1) # [bs, 1, state_dim]
        if self.node.type + '/node_history_encoder' not in self.curr_hidden_states:
            outputs, self.curr_hidden_states[self.node.type + '/node_history_encoder'] = self.node_modules[self.node.type + '/node_history_encoder'](new_state)
        else:
            outputs, self.curr_hidden_states[self.node.type + '/node_history_encoder'] = self.node_modules[self.node.type + '/node_history_encoder'](new_state, self.curr_hidden_states[self.node.type + '/node_history_encoder'])

        outputs = F.dropout(outputs,
                    p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                    training=False) # [bs, max_time, enc_rnn_dim]

        return outputs[:, 0, :]


    def encode_edge(self, edge_type, connected_nodes, new_inputs_dict):
        input_feature_list = [new_inputs_dict[node] for node in (list(connected_nodes) + self.removed_neighbors_by_edge_type[edge_type])]
        stacked_edge_states = torch.unsqueeze(torch.stack(input_feature_list, dim=0), dim=1)

        if self.dynamic_edges == 'yes':
            edge_mask = torch.stack([self.get_mask_for_edge_to(other_node)
                                        for other_node in (list(connected_nodes) + self.removed_neighbors_by_edge_type[edge_type])],
                                    dim=0)

        if self.edge_state_combine_method == 'sum':
            # Used in Structural-RNN to combine edges as well.
            combined_neighbors = torch.sum(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be a scalar.
                edge_mask = torch.clamp(torch.sum(edge_mask, dim=0, keepdim=True), max=1.)

        elif self.edge_state_combine_method == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            combined_neighbors = torch.max(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be a scalar.
                edge_mask = torch.clamp(torch.max(edge_mask, dim=0, keepdim=True), max=1.)

        elif self.edge_state_combine_method == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            combined_neighbors = torch.mean(stacked_edge_states, dim=0)
            if self.dynamic_edges == 'yes':
                # Should now be a scalar.
                edge_mask = torch.clamp(torch.mean(edge_mask, dim=0, keepdim=True), max=1.)

        joint_history = torch.cat([combined_neighbors, torch.unsqueeze(new_inputs_dict[self.node], dim=0)], dim=2)
        # joint_history = F.dropout(joint_history,
        #                           p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
        #                           training=(mode == ModeKeys.TRAIN))
        if edge_type + '/edge_encoder' not in self.curr_hidden_states:
            outputs, self.curr_hidden_states[edge_type + '/edge_encoder'] = self.node_modules[edge_type + '/edge_encoder'](joint_history)
        else:
            outputs, self.curr_hidden_states[edge_type + '/edge_encoder'] = self.node_modules[edge_type + '/edge_encoder'](joint_history, self.curr_hidden_states[edge_type + '/edge_encoder'])

        outputs = F.dropout(outputs,
                            p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=False) # [bs, max_time, enc_rnn_dim]

        if self.dynamic_edges == 'yes':
            return outputs[:, 0, :] * edge_mask
        else:
            return outputs[:, 0, :]   # [bs, enc_rnn_dim]


    def encode_total_edge_influence(self, encoded_edges, node_history_encoder, batch_size):
        if self.edge_influence_combine_method == 'sum':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.edge_influence_combine_method == 'max':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.edge_influence_combine_method == 'bi-rnn':
            if len(encoded_edges) == 0:
                # Four times because we're trying to mimic a bi-directional
                # RNN's output (which is c and h from both ends).
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                # encoded_edges = F.dropout(encoded_edges,
                #                   p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                #                   training=(mode == ModeKeys.TRAIN))
                if self.node.type + '/edge_influence_encoder' not in self.curr_hidden_states:
                    _, self.curr_hidden_states[self.node.type + '/edge_influence_encoder'] = self.node_modules[self.node.type + '/edge_influence_encoder'](encoded_edges)
                else:
                    _, self.curr_hidden_states[self.node.type + '/edge_influence_encoder'] = self.node_modules[self.node.type + '/edge_influence_encoder'](encoded_edges, self.curr_hidden_states[self.node.type + '/edge_influence_encoder'])

                combined_edges = unpack_RNN_state(self.curr_hidden_states[self.node.type + '/edge_influence_encoder'])
                combined_edges = F.dropout(combined_edges,
                                  p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                  training=False)

        elif self.edge_influence_combine_method == 'attention':
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[self.node.type + '/edge_influence_encoder'](encoded_edges, node_history_encoder)
                combined_edges = F.dropout(combined_edges,
                                  p=1.-self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                  training=False)

        return combined_edges


    # robot_future is optional here since you can use the same one from encoder_forward,
    # but if it's given then we'll re-run that part of the model (if the node is adjacent to the robot).
    def decoder_forward(self, num_predicted_timesteps, num_samples, robot_future=None, most_likely=False):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        if self.robot_node is not None and robot_future is not None:
            self.TD["x"] = self.create_encoder_rep(self.TD, robot_future)
            self.latent.p_dist = self.p_z_x(self.TD["x"], mode)

        z = self.latent.sample_p(num_samples, mode, most_likely=most_likely)
        y_dist, our_sampled_future = self.p_y_xz(self.TD["x"], z, self.TD, mode,
                                                 num_predicted_timesteps,
                                                 num_samples,
                                                 most_likely=most_likely) # y_dist.mean is [k, bs, ph*state_dim]

        predictions_dict = {str(self.node) + "/y": our_sampled_future,
                            str(self.node) + "/z": z}
        return predictions_dict
