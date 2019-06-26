import numpy as np
import torch
from model.node_model import MultimodalGenerativeCVAE
from model.model_utils import ModeKeys, exp_anneal
from stg_node import STGNode, convert_to_label_node, convert_from_label_node
from utils import plot_utils

# How to handle removal of edges:
# Cosine weight the previous output to 0.0 weight over a few timesteps, 
# then remove that LSTM from computations.

# How to handle addition of edges:
# Create a new LSTM with zero init hidden state and cosine weight 
# the added points up to 1.0 weight over a few timesteps.
# This gating is on the output of the LSTM.
class SpatioTemporalGraphCVAEModel(object):
    def __init__(self, robot_node, model_registrar,
                 hyperparams, kwargs_dict, log_writer,
                 device):
        super(SpatioTemporalGraphCVAEModel, self).__init__()
        self.hyperparams = hyperparams
        self.edge_state_combine_method = kwargs_dict['edge_state_combine_method']
        self.edge_influence_combine_method = kwargs_dict['edge_influence_combine_method']
        self.dynamic_edges = kwargs_dict['dynamic_edges']
        self.robot_node = robot_node
        self.log_writer = log_writer
        self.device = device

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()


    def set_scene_graph(self, scene_graph):    
        self.node_models_dict.clear()

        for node in scene_graph.nodes:
            if node != self.robot_node:
                self.nodes.add(node)
                kwargs_dict = {'edge_state_combine_method': self.edge_state_combine_method,
                               'edge_influence_combine_method': self.edge_influence_combine_method,
                               'dynamic_edges': self.dynamic_edges,
                               'hyperparams': self.hyperparams}

                self.node_models_dict[str(node)] = MultimodalGenerativeCVAE(node, 
                                                                            self.model_registrar,
                                                                            self.robot_node,
                                                                            kwargs_dict,
                                                                            self.device,
                                                                            scene_graph=scene_graph,
                                                                            log_writer=self.log_writer)


    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)


    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()


    def step_annealers(self):
        for node in self.node_models_dict:
            self.node_models_dict[node].step_annealers()


    def train_loss(self, inputs, labels, num_predicted_timesteps):
        mode = ModeKeys.TRAIN
        inputs, labels = self.standardize(mode, inputs, labels)

        # This is important to ensure that each node model is using the same training data points.
        mhl, ph = self.hyperparams['minimum_history_length'], self.hyperparams['prediction_horizon']
        if np.any(inputs['traj_lengths'] < mhl + ph):
            batch_size = inputs['traj_lengths'].shape[0]
            idxs_to_keep = [batch_idx for batch_idx, traj_len in enumerate(inputs['traj_lengths']) if traj_len >= mhl + ph]

            for key, value in inputs.items():
                inputs[key] = value[idxs_to_keep]

            for key, value in labels.items():
                labels[key] = value[idxs_to_keep]

            print("""WARNING: There are trajectory lengths less than %d (= minimum_history_length + prediction_horizon) in the training input!
Ignoring those indices, the batch size will be reduced from %d to %d.""" % (mhl + ph, batch_size, inputs['traj_lengths'].shape[0]))

        traj_lengths = inputs['traj_lengths']
        prediction_timesteps = mhl - 1 + torch.fmod(torch.randint(low=0, 
                                                                  high=2**31-1, 
                                                                  size=traj_lengths.shape).to(self.device),
                                                    traj_lengths-mhl-ph+1).long()

        losses = list()
        for node in self.nodes:
            model = self.node_models_dict[str(node)]
            losses.append(model.train_loss(inputs, 
                                           labels[convert_to_label_node(node)],
                                           num_predicted_timesteps,
                                           prediction_timesteps))

        mean_loss = torch.mean(torch.stack(losses))
        return mean_loss


    def eval_loss(self, orig_inputs, orig_labels, num_predicted_timesteps):
        mode = ModeKeys.EVAL
        inputs, labels = self.standardize(mode, orig_inputs, orig_labels)

        # This is important to ensure that each node model is using the same eval data points.
        mhl, ph = self.hyperparams['minimum_history_length'], self.hyperparams['prediction_horizon']
        if np.any(inputs['traj_lengths'] < mhl + ph):
            batch_size = inputs['traj_lengths'].shape[0]
            idxs_to_keep = [batch_idx for batch_idx, traj_len in enumerate(inputs['traj_lengths']) if traj_len >= mhl + ph]

            for key, value in inputs.items():
                inputs[key] = value[idxs_to_keep]

            for key, value in labels.items():
                labels[key] = value[idxs_to_keep]

            print("""WARNING: There are trajectory lengths less than %d (= minimum_history_length + prediction_horizon) in the evaluation input!
Ignoring those indices, the batch size will be reduced from %d to %d.""" % (mhl + ph, batch_size, inputs['traj_lengths'].shape[0]))

        nll_q_is_values = list()
        nll_p_values = list()
        nll_exact_values = list()
        for node in self.nodes:
            model = self.node_models_dict[str(node)]
            (nll_q_is, nll_p, nll_exact) = model.eval_loss(inputs,
                                                           labels[convert_to_label_node(node)],
                                                           num_predicted_timesteps)
            nll_q_is_values.append(nll_q_is)
            nll_p_values.append(nll_p)
            nll_exact_values.append(nll_exact)

        nll_q_is, nll_p, nll_exact = torch.mean(torch.stack(nll_q_is_values)), torch.mean(torch.stack(nll_p_values)), torch.mean(torch.stack(nll_exact_values))
        return nll_q_is, nll_p, nll_exact


    def predict(self, inputs, num_predicted_timesteps, num_samples, most_likely=False):
        mode = ModeKeys.PREDICT
        inputs = self.standardize(mode, inputs)

        predictions_dict = dict()
        for node in self.nodes:
            model = self.node_models_dict[str(node)]
            output_dict = model.predict(inputs, num_predicted_timesteps, num_samples, most_likely=most_likely)

            node_mean = self.hyperparams['nodes_standardization'][node]['mean'][self.hyperparams['pred_indices']]
            node_std = self.hyperparams['nodes_standardization'][node]['std'][self.hyperparams['pred_indices']]
            output_dict[str(node) + "/y"] = self.unstandardize(output_dict[str(node) + "/y"], node_mean, node_std)
            predictions_dict.update(output_dict)

        return predictions_dict


    def standardize_fn(self, tensor, mean, std):
        return (tensor - mean) / std


    def standardize(self, mode, inputs, labels=None):
        features_standardized = dict()

        if 'traj_lengths' in inputs:
            features_standardized['traj_lengths'] = inputs['traj_lengths']

        if 'edge_scaling_mask' in inputs:
            features_standardized['edge_scaling_mask'] = inputs['edge_scaling_mask']

        for node in inputs:
            if isinstance(node, STGNode) or (mode == ModeKeys.PREDICT and node == str(self.robot_node) + '_future'):
                if mode == ModeKeys.PREDICT and node == str(self.robot_node) + '_future':
                    # This is handling the case of normalizing the future robot actions, 
                    # which really should just take the same normalization as the robot
                    # from training.
                    node_mean = self.hyperparams['nodes_standardization'][self.robot_node]['mean']
                    node_std = self.hyperparams['nodes_standardization'][self.robot_node]['std']
                else:
                    node_mean = self.hyperparams['nodes_standardization'][node]['mean']
                    node_std = self.hyperparams['nodes_standardization'][node]['std']

                node_mean = torch.from_numpy(node_mean).float().to(self.device)
                node_std = torch.from_numpy(node_std).float().to(self.device)

                features_standardized[node] = self.standardize_fn(inputs[node], node_mean, node_std)
                if mode == ModeKeys.TRAIN and self.hyperparams['fuzz_factor'] > 0:
                    features_standardized[node] += self.hyperparams['fuzz_factor']*torch.randn_like(features_standardized[node])

        if labels is None:
            return features_standardized

        labels_standardized = dict()
        for label in labels:
            if isinstance(label, STGNode):
                label_node = convert_from_label_node(label)
                node_mean = self.hyperparams['nodes_standardization'][label_node]['mean'][self.hyperparams['pred_indices']]
                node_std = self.hyperparams['nodes_standardization'][label_node]['std'][self.hyperparams['pred_indices']]

                node_mean = torch.from_numpy(node_mean).float().to(self.device)
                node_std = torch.from_numpy(node_std).float().to(self.device)

                labels_standardized[label] = self.standardize_fn(labels[label], node_mean, node_std)
                if mode == ModeKeys.TRAIN and self.hyperparams['fuzz_factor'] > 0:
                    labels_standardized[label] += self.hyperparams['fuzz_factor']*torch.randn_like(labels_standardized[label])

        return features_standardized, labels_standardized


    def unstandardize(self, output, labels_mean, labels_std):
        torch_std = torch.from_numpy(labels_std).float().to(self.device)
        torch_mean = torch.from_numpy(labels_mean).float().to(self.device)
        return output * torch_std + torch_mean
