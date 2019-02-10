import torch
from model.dyn_stg import SpatioTemporalGraphCVAEModel
from model.online_node_model import OnlineMultimodalGenerativeCVAE
from model.model_utils import ModeKeys
from utils.scene_utils import SceneGraph
from stg_node import STGNode

# How to handle removal of edges:
# Cosine weight the previous output to 0.0 weight over a few timesteps, 
# then remove that LSTM from computations.

# How to handle addition of edges:
# Create a new LSTM with zero init hidden state and cosine weight 
# the added points up to 1.0 weight over a few timesteps.
# This gating is on the output of the LSTM.
class OnlineSpatioTemporalGraphCVAEModel(SpatioTemporalGraphCVAEModel):
    def __init__(self, robot_node, model_registrar,
                 hyperparams, kwargs_dict, device):
        super(OnlineSpatioTemporalGraphCVAEModel, self).__init__(robot_node, 
            model_registrar, hyperparams, kwargs_dict, 
            log_writer=None, device=device)

        self.edge_addition_filter = kwargs_dict['edge_addition_filter']
        self.edge_removal_filter = kwargs_dict['edge_removal_filter']


    def _add_node_model(self, node):
        if node in self.nodes:
            raise ValueError('%s was already added to this graph!' % str(node))

        if node == self.robot_node:
            # We don't want to model the robot node (we are the robot).
            return

        self.nodes.add(node)
        kwargs_dict = {'edge_state_combine_method': self.edge_state_combine_method,
                       'edge_influence_combine_method': self.edge_influence_combine_method,
                       'dynamic_edges': self.dynamic_edges,
                       'edge_addition_filter': self.edge_addition_filter,
                       'edge_removal_filter': self.edge_removal_filter,
                       'hyperparams': self.hyperparams}

        self.node_models_dict[str(node)] = OnlineMultimodalGenerativeCVAE(node, 
                                                                    self.model_registrar,
                                                                    self.robot_node,
                                                                    kwargs_dict,
                                                                    self.device,
                                                                    self.scene_graph)


    def _remove_node_model(self, node):
        if node not in self.nodes:
            raise ValueError('%s is not in this graph!' % str(node))

        self.nodes.remove(node)
        del self.node_models_dict[str(node)]


    def set_scene_graph(self, scene_graph): 
        self.scene_graph = scene_graph   
        self.node_models_dict.clear()

        for node in scene_graph.active_nodes:
            if node != self.robot_node:
                self._add_node_model(node)


    def standardize(self, mode, inputs):
        features_standardized = dict()
        if 'traj_lengths' in inputs:
            features_standardized['traj_lengths'] = inputs['traj_lengths']

        for node in inputs:
            if isinstance(node, STGNode) or (mode == ModeKeys.PREDICT and node == str(self.robot_node) + '_future'):
                if node == str(self.robot_node) + '_future':
                    # This is handling the case of normalizing the future robot actions, 
                    # which really should just take the same normalization as the robot
                    # from training.
                    node_mean = self.hyperparams['nodes_standardization'][self.robot_node]['mean']
                    node_std = self.hyperparams['nodes_standardization'][self.robot_node]['std']
                else:
                    node_mean = self.hyperparams['nodes_standardization'][node]['mean']
                    node_std = self.hyperparams['nodes_standardization'][node]['std']

                features_standardized[node] = self.standardize_fn(torch.from_numpy(inputs[node]).float().to(self.device), node_mean, node_std)

        return features_standardized


    def unstandardize(self, output, labels_mean, labels_std):
        return output * labels_std + labels_mean


    def incremental_forward(self, robot_future, new_pos_dict, new_inputs_dict, 
                            num_predicted_timesteps, num_samples):
        # This is the magic function which should make this model
        # able to handle streams. Thank you PyTorch!!
        # The way this function works is by appending the new datapoints to the 
        # ends of each of the LSTMs in the graph. Then, we recalculate the 
        # encoder's output vector h_x and feed that into the decoder to sample new outputs.
        mode = ModeKeys.PREDICT
        
        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            new_scene_graph = SceneGraph()
            new_scene_graph.create_from_scene_dict(new_pos_dict, self.scene_graph.edge_radius)

            new_inputs_dict = self.standardize(mode, new_inputs_dict)
            robot_future_std = self.standardize(mode, {str(self.robot_node) + '_future': robot_future})[str(self.robot_node) + '_future']

            if self.dynamic_edges == 'yes':
                new_nodes, removed_nodes, new_neighbors, removed_neighbors = new_scene_graph - self.scene_graph
                
                # Aside from updating the scene graph, this for loop updates the graph model 
                # structure of all affected nodes.
                not_removed_nodes = [node for node in self.nodes if node not in removed_nodes]
                self.scene_graph = new_scene_graph
                for node in not_removed_nodes:
                    self.node_models_dict[str(node)].update_graph(new_scene_graph, new_neighbors, removed_neighbors)

                # These next 2 for loops add or remove entire node models.
                for node in new_nodes:
                    self._add_node_model(node)

                for node in removed_nodes:
                    self._remove_node_model(node)

            # This actually updates the node models with the newly observed data.
            for node in self.node_models_dict:
                self.node_models_dict[str(node)].encoder_forward(new_inputs_dict, robot_future_std)

            # If num_predicted_timesteps or num_samples == 0 then do not run the decoder at all, 
            # just update the encoder LSTMs.
            if num_predicted_timesteps == 0 or num_samples == 0:
                return

            assert robot_future.shape[0] == num_predicted_timesteps

        return self.sample_model(num_predicted_timesteps, num_samples)
    

    def sample_model(self, num_predicted_timesteps, num_samples, robot_future=None):
        # Just start from the encoder output (minus the 
        # robot future) and get num_samples of 
        # num_predicted_timesteps-length trajectories.
        if num_predicted_timesteps == 0 or num_samples == 0:
            return

        mode = ModeKeys.PREDICT

        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            if robot_future is not None: 
                assert robot_future.shape[0] == num_predicted_timesteps
                robot_future = self.standardize(mode, {str(self.robot_node) + '_future': robot_future})

            predictions_dict = dict()
            for node in self.nodes:
                model = self.node_models_dict[str(node)]
                output_dict = model.decoder_forward(num_predicted_timesteps, num_samples, robot_future)

                node_mean = self.hyperparams['nodes_standardization'][node]['mean'][self.hyperparams['pred_indices']]
                node_std = self.hyperparams['nodes_standardization'][node]['std'][self.hyperparams['pred_indices']]
                output_dict[str(node) + "/y"] = self.unstandardize(output_dict[str(node) + "/y"], node_mean, node_std)
                predictions_dict.update(output_dict)

        return predictions_dict


    def forward(self, pos_history, input_history, robot_future,
                num_predicted_timesteps, num_samples):
        # This is the standard forward prediction function, 
        # if you have some historical data and just want to 
        # predict forward some number of timesteps.
        for i in range(input_history.values()[0].shape[0]):
            self.incremental_forward(robot_future,
                                     {k: v[i] for k, v in pos_history.items()},
                                     {k: v[i] for k, v in input_history.items()},
                                     num_predicted_timesteps=0,
                                     num_samples=0)

        return self.sample_model(num_predicted_timesteps,
                                 num_samples)
