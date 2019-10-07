import numpy as np
from itertools import product
from .scene_graph import SceneGraph


class Scene(object):
    def __init__(self, type_enum=None, timesteps=0, dt=1):
        self.type_enum = type_enum
        self.timesteps = timesteps
        self.dt = dt

        self.nodes = []

        self.robot = None

        self.standardization = {}

        self.last_query_present_nodes_t = None
        self.last_query_present_nodes_n = None

    def length(self):
        return self.timesteps * self.dt

    def present_nodes(self, timesteps, type=None, min_history_timesteps=0, min_future_timesteps=0):
        if (self.last_query_present_nodes_t == timesteps).all():
            return self.last_query_present_nodes_n
        present_nodes = {}
        for node in self.nodes:
            if type is None or node.type == type:
                lower_bound = timesteps - min_history_timesteps
                upper_bound = timesteps + min_future_timesteps
                mask = (node.first_timestep <= lower_bound) & (upper_bound <= node.last_timestep)
                if mask.any():
                    timestep_indices_present = np.nonzero(mask)[0]
                    for timestep_index_present in timestep_indices_present:
                        if timesteps[timestep_index_present] in present_nodes.keys():
                            present_nodes[timesteps[timestep_index_present]].append(node)
                        else:
                            present_nodes[timesteps[timestep_index_present]] = [node]

        self.last_query_present_nodes_t = timesteps
        self.last_query_present_nodes_n = present_nodes
        return present_nodes

    def sample_timesteps(self, batch_size):
        if batch_size > self.timesteps:
            batch_size = self.timesteps
        return np.random.choice(np.arange(0, self.timesteps), size=batch_size, replace=False)

    def get_scene_graph(self, timestep, edge_radius, dims, edge_addition_filter_l=0, edge_removal_filter_l=0):
        timestep_range = np.array([timestep - edge_addition_filter_l, timestep + edge_removal_filter_l])
        node_pos_dict = dict()
        present_nodes = self.present_nodes(np.array([timestep]))

        for node in present_nodes[timestep]:
            node_pos_dict[node] = np.squeeze(node.get(timestep_range, 'position', dims=dims))
        sg = SceneGraph.create_from_temp_scene_dict(node_pos_dict,
                                                    edge_radius,
                                                    duration=edge_addition_filter_l + edge_removal_filter_l + 1)
        return sg

    def get_edge_types(self):
        edge_prod = product([node_type.name for node_type in self.type_enum], repeat=2)
        edge_types = []
        for edge_comb in edge_prod:
            edge_types.append('-'.join(edge_comb))
        return edge_types

    def get_standardize_params(self, entities, dims):
        standardize_mean_list = list()
        standardize_std_list = list()
        for entity in entities:
            for dim in dims:
                standardize_mean_list.append(self.standardization[entity][dim]['mean'])
                standardize_std_list.append(self.standardization[entity][dim]['std'])
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

        return standardize_mean, standardize_std

    def standardize(self, array, entities, dims):
        mean, std = self.get_standardize_params(entities, dims)
        return np.where(np.isnan(array), np.array(np.nan), (array - mean) / std)

    def unstandardize(self, array, entities, dims):
        mean, std = self.get_standardize_params(entities, dims)
        return array * std + mean

    def __repr__(self):
        return f"Scene: Duration: {self.length()}s," \
               f" Nodes: {len(self.nodes)}."
