import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.signal as ss
from collections import defaultdict
import warnings


class SceneGraph(object):
    def __init__(self, edge_radius, nodes=None, adj_cube=np.zeros((1, 0, 0)), t=0):
        self.edge_radius = edge_radius
        self.nodes = nodes
        if nodes is None:
            self.nodes = list()
        self.adj_cube = adj_cube
        self.current_t = t
        self.neigbors_via_edge_type_all_t = None

    @property
    def neigbors_via_edge_type(self):
        if self.neigbors_via_edge_type_all_t is None:
            self.neigbors_via_edge_type_all_t = self.get_neigbors_via_edge_type()
        return self.neigbors_via_edge_type_all_t

    def get_neigbors_via_edge_type(self, t=None):
        node_edges_and_neighbors = {node: defaultdict(set) for node in self.nodes}
        edge_types = defaultdict(list)
        if t is None:
            adj_mat = np.max(self.adj_cube, axis=0)
        else:
            adj_mat = self.adj_cube[t]
        for i, curr_node in enumerate(self.nodes):
            for j, curr_neighbor in enumerate(self.nodes):
                if adj_mat[i, j] == 1:
                    edge_type = self.get_edge_type(curr_node, curr_neighbor)
                    edge_types[curr_node].append(edge_type)
                    node_edges_and_neighbors[curr_node][edge_type].add(curr_neighbor)
        return node_edges_and_neighbors

    def get_num_edges(self, t=0):
        return np.sum(self.adj_cube[t]) // 2

    def get_index(self, node):
        return list(self.nodes).index(node)

    @staticmethod
    def get_edge_type(n1, n2):
        return '-'.join(sorted([str(n1), str(n2)]))

    @classmethod
    def create_from_temp_scene_dict(cls, scene_temp_dict, edge_radius, duration=1, t=0):
        """
        Construct a spatiotemporal graph from agent positions in a dataset.

        returns: sg: An aggregate SceneGraph of the dataset.
        """
        nodes = scene_temp_dict.keys()
        N = len(nodes)
        total_timesteps = duration

        position_cube = np.zeros((total_timesteps, N, 2))

        adj_cube = np.zeros((total_timesteps, N, N), dtype=np.int8)

        for node_idx, node in enumerate(nodes):
            position_cube[:, node_idx] = scene_temp_dict[node]

        agg_adj_matrix = np.zeros((N, N), dtype=np.int8)

        for timestep in range(position_cube.shape[0]):
            dists = squareform(pdist(position_cube[timestep], metric='euclidean'))

            # Put a 1 for all agent pairs which are closer than the edge_radius.
            # Can produce a warning as dists can be nan if no data for node is available.
            # This is accepted as nan <= x evaluates to False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adj_matrix = (dists <= edge_radius).astype(np.int8)

            # Remove self-loops.
            np.fill_diagonal(adj_matrix, 0)

            agg_adj_matrix |= adj_matrix

            adj_cube[timestep] = adj_matrix

        sg = cls(edge_radius, nodes, adj_cube, t)
        return sg

    def get_edge_scaling(self, t, edge_addition_filter, edge_removal_filter, node=None):
        new_edges = np.minimum(ss.fftconvolve(self.adj_cube,
                                              np.reshape(edge_addition_filter, (-1, 1, 1)), 'full'), 1.)[t]
        old_edges = np.minimum(ss.fftconvolve(self.adj_cube,
                                              np.reshape(edge_removal_filter, (-1, 1, 1)), 'full'), 1.)[t]

        adj_mat = np.max(self.adj_cube, axis=0)

        edge_scaling = np.minimum(new_edges + old_edges, 1.)

        if node is None:
            return edge_scaling
        else:
            node_index = self.get_index(node)
            return edge_scaling[
                node_index, adj_mat[node_index] > 0.]  # We only want nodes which were connected at some point

    def get_adj_matrix(self):
        N = len(self.scene_dict)

        if N == 0:
            return None, list()

        active_idxs = list()

        pos_matrix = np.empty((N, 2))
        for idx, node in enumerate(self.scene_dict):
            #     x position   ,     y position
            (pos_matrix[idx][0], pos_matrix[idx][1]) = self.scene_dict[node]

            if np.asarray(self.scene_dict[node]).any():
                active_idxs.append(idx)

        dists = squareform(pdist(pos_matrix, metric='euclidean'))

        # Put a 1 for all agent pairs which are closer than the edge_radius.
        adj_matrix = (dists <= self.edge_radius).astype(int)
        assert len(adj_matrix.shape) == 2 and adj_matrix.shape == (N, N)

        # Remove self-loops.
        np.fill_diagonal(adj_matrix, 0)

        return adj_matrix, active_idxs

    def get_st_graph_info(self):
        """Construct a spatiotemporal graph from N agent positions.

        returns: nodes: An N-length list of ordered nodes.
                 edge_types: An N-size dict containing lists of edge-type string
                             names per node.
                 node_edges_and_neighbors: An N-size dict of edge-types per node,
                                           as well as which nodes are neighboring
                                           along edges of that type.
        """
        N = len(self.scene_dict)

        if N == 0:
            return list(), defaultdict(list), dict()

        nodes = list(self.scene_dict.keys())

        adj_matrix, active_idxs = self.get_adj_matrix()
        assert adj_matrix.shape == (N, N)

        node_edges_and_neighbors = {node: defaultdict(set) for node in nodes}
        edge_types = defaultdict(list)
        for i in active_idxs:
            curr_node = nodes[i]
            for j in active_idxs:
                curr_neighbor = nodes[j]
                if adj_matrix[i, j] == 1:
                    edge_type = self.get_edge_type(curr_node, curr_neighbor)
                    edge_types[curr_node].append(edge_type)

                    node_edges_and_neighbors[curr_node][edge_type].add(curr_neighbor)

        return nodes, edge_types, node_edges_and_neighbors
