# import sys, os
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from stg_node import STGNode
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.signal as ss
from collections import defaultdict
import matplotlib.pyplot as plt


def get_edge_id(node_A, node_B):
    return '-'.join(sorted([str(node_A), str(node_B)]))


def get_edge_type(node_A, node_B):
    return '-'.join(sorted([node_A.type, node_B.type]))


class Scene(object):
    def __init__(self, agent_xys):
        self.scene_dict = dict()
        for node, pos in agent_xys.items():
            self.add_agent((node, pos))


    def add_agent(self, new_agent):
        node, pos = new_agent

        node_parts = node.split('/')
        node_name = node_parts[-1]
        node_type = '/'.join(node_parts[:-1])

        new_node = STGNode(node_name, node_type)
        self.scene_dict[new_node] = pos


    def get_graph(self, edge_radius):
        scene_graph = SceneGraph()
        scene_graph.create_from_scene_dict(self.scene_dict, edge_radius)
        return scene_graph


    def visualize(self, ax, radius=0.3, circle_edge_width=0.5):
        for node in self.scene_dict:
            # Current Node Position
            circle = plt.Circle(xy=(self.scene_dict[node][0],
                                    self.scene_dict[node][1]),
                                radius=radius,
                                facecolor='grey',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

            ax.text(self.scene_dict[node][0] + radius + 0.1,
                    self.scene_dict[node][1],
                    node.name,
                    zorder=4)


class DirectionalEdge(object):
    def __init__(self, curr_node, other_node):
        self.id = get_edge_id(curr_node, other_node)
        self.type = get_edge_type(curr_node, other_node)
        self.curr_node = curr_node
        self.other_node = other_node


    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.id == other.id)


    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        return hash(self.id)


    def __repr__(self):
        return self.id


class SceneGraph(object):
    def __init__(self):
        self.edge_scaling_mask = None


    def __sub__(self, other):
        new_nodes = [node for node in self.active_nodes if node not in other.active_nodes]
        removed_nodes = [node for node in other.active_nodes if node not in self.active_nodes]

        new_neighbors = defaultdict(dict)
        for node, edges_and_neighbors in self.node_edges_and_neighbors.items():
            if node in other.node_edges_and_neighbors:
                for edge_type, neighbors in edges_and_neighbors.items():
                    if edge_type in other.node_edges_and_neighbors[node]:
                        new_items = list(self.node_edges_and_neighbors[node][edge_type] - other.node_edges_and_neighbors[node][edge_type])
                        if len(new_items) > 0:
                            new_neighbors[node][edge_type] = new_items
                    else:
                        new_neighbors[node][edge_type] = self.node_edges_and_neighbors[node][edge_type]
            else:
                new_neighbors[node] = self.node_edges_and_neighbors[node]

        removed_neighbors = defaultdict(dict)
        for node, edges_and_neighbors in other.node_edges_and_neighbors.items():
            if node in self.node_edges_and_neighbors:
                for edge_type, neighbors in edges_and_neighbors.items():
                    if edge_type in self.node_edges_and_neighbors[node]:
                        removed_items = list(other.node_edges_and_neighbors[node][edge_type] - self.node_edges_and_neighbors[node][edge_type])
                        if len(removed_items) > 0:
                            removed_neighbors[node][edge_type] = removed_items
                    else:
                        removed_neighbors[node][edge_type] = other.node_edges_and_neighbors[node][edge_type]
            else:
                removed_neighbors[node] = other.node_edges_and_neighbors[node]

        # Cleaning up *_neighbors because of the influence of new and removed nodes.
        # This is because add_node_model in online_dyn_stg will already populate
        # the new model with the correct, new edges since the new scene graph contains them.
        for node in new_nodes:
            del new_neighbors[node]

        for node in removed_nodes:
            del removed_neighbors[node]

        return new_nodes, removed_nodes, new_neighbors, removed_neighbors


    def create_from_adj_matrix(self, adj_matrix, nodes, edge_radius,
                               adj_cube=None, inactive_nodes=list()):
        """Populates the SceneGraph instance from an adjacency matrix.

        adj_matrix: N x N adjacency matrix.
        nodes; N-length list containing the node objects
        edge_radius: float describing the radius used to generate the adjacency matrix.
        """
        self.adj_matrix = adj_matrix
        self.edge_radius = edge_radius
        self.nodes = nodes
        self.adj_cube = adj_cube

        N = len(nodes)

        node_edges_and_neighbors = {node: defaultdict(set) for node in nodes}
        edge_types = defaultdict(list)
        for i in range(N):
            curr_node = nodes[i]
            for j in range(N):
                curr_neighbor = nodes[j]
                if adj_matrix[i, j] == 1:
                    edge_type = get_edge_type(curr_node, curr_neighbor)
                    edge_types[curr_node].append(edge_type)

                    node_edges_and_neighbors[curr_node][edge_type].add(curr_neighbor)

        self.edge_types = edge_types
        self.node_edges_and_neighbors = node_edges_and_neighbors

        self.active_nodes = [node for node in self.nodes if node not in inactive_nodes]

        active_idxs = [self.nodes.index(node) for node in self.active_nodes]
        self.num_edges = 0
        for idx, node in enumerate(self.active_nodes):
            self.num_edges += np.sum(self.adj_matrix[active_idxs[idx], active_idxs])
        self.num_edges /= 2


    def create_from_scene_dict(self, scene_dict, edge_radius, adj_cube=None):
        """Populates the SceneGraph instance from a dictionary describing a scene.

        scene_dict: N x 2 dict describing the current x and y position of each agent.
        edge_radius: float describing the radius around a node that defines edge creation.
        """
        self.edge_radius = edge_radius
        self.scene_dict = scene_dict
        self.nodes, self.edge_types, self.node_edges_and_neighbors = self.get_st_graph_info()
        self.adj_matrix, active_idxs = self.get_adj_matrix()
        self.adj_cube = adj_cube

        self.active_nodes = [self.nodes[idx] for idx in active_idxs]

        self.num_edges = 0
        for idx, node in enumerate(self.active_nodes):
            self.num_edges += np.sum(self.adj_matrix[active_idxs[idx], active_idxs])
        self.num_edges /= 2


    def __eq__(self, other):
        return self.adj_matrix == other.adj_matrix


    def __ne__(self, other):
        return self.adj_matrix != other.adj_matrix


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
        pos_matrix = np.array(list(self.scene_dict.values()))
        assert pos_matrix.shape == (N, 2)

        adj_matrix, active_idxs = self.get_adj_matrix()
        assert adj_matrix.shape == (N, N)

        node_edges_and_neighbors = {node: defaultdict(set) for node in nodes}
        edge_types = defaultdict(list)
        for i in active_idxs:
            curr_node = nodes[i]
            for j in active_idxs:
                curr_neighbor = nodes[j]
                if adj_matrix[i, j] == 1:
                    edge_type = get_edge_type(curr_node, curr_neighbor)
                    edge_types[curr_node].append(edge_type)

                    node_edges_and_neighbors[curr_node][edge_type].add(curr_neighbor)

        return nodes, edge_types, node_edges_and_neighbors


    def compute_edge_scaling(self, edge_addition_filter, edge_removal_filter):
        if self.adj_cube is None:
            return

        # (data_id, time, N, N)
        new_edges = np.minimum(ss.fftconvolve(self.adj_cube, np.reshape(edge_addition_filter, (1, -1, 1, 1)), 'full'), 1.)[:, (len(edge_addition_filter) - 1):]
        old_edges = np.minimum(ss.fftconvolve(self.adj_cube, np.reshape(edge_removal_filter, (1, -1, 1, 1)), 'full'), 1.)[:, :-(len(edge_removal_filter) - 1)]
        self.edge_scaling_mask = np.minimum(new_edges + old_edges, 1.)


    def render(self, pos_matrix,
               filename='graph_video.mp4'):
        """
        Render a spatiotemporal graph video from N agent positions.

        pos_matrix: T x N x 2 matrix describing the x and y positions
                    of each agent over time.
        """
        import matplotlib.lines as mlines
        from PIL import Image
        import imageio
        from cStringIO import StringIO

        fig, ax = plt.subplots()
        ax.set_xlim(left=np.nanmin(pos_matrix[:, :, 0]) - 1, right=np.nanmax(pos_matrix[:, :, 0]) + 1)
        ax.set_ylim(bottom=np.nanmin(pos_matrix[:, :, 1]) - 1, top=np.nanmax(pos_matrix[:, :, 1]) + 1)
        ax.set_xlabel('Longitudinal Court Position (ft)')
        ax.set_ylabel('Lateral Court Position (ft)')
        l, = plt.plot([], [], marker='o', color='white', markeredgecolor='k', markerfacecolor='white', markeredgewidth=1.0, zorder=3)

        # Get adj_matrix from each timestep.
        images = list()
        for t in xrange(pos_matrix.shape[0]):
            adj_matrix, active_idxs = get_adj_matrix_helper(pos_matrix[t], self.edge_radius, self.nodes)
            N = adj_matrix.shape[0]

            # Edges
            lines = []
            for agent1 in active_idxs:
                for agent2 in active_idxs:
                    if adj_matrix[agent1, agent2] == 1:
                        line = mlines.Line2D([pos_matrix[t, agent1, 0], pos_matrix[t, agent2, 0]],
                                             [pos_matrix[t, agent1, 1], pos_matrix[t, agent2, 1]],
                                             color='k')
                        ax.add_line(line)
                        lines.append(line)

            # Nodes
            new_data = np.ones((pos_matrix.shape[1]*2, 2))*np.nan
            new_data[::2] = pos_matrix[t, :, 0:2]
            l.set_data(new_data[:, 0], new_data[:, 1])

            buffer_ = StringIO()
            plt.savefig(buffer_, format = "png", dpi=150)
            buffer_.seek(0)

            data = np.asarray(Image.open( buffer_ ))

            images.append(data)

            for line in lines:
                line.remove()

        imageio.mimsave(filename, images, fps=15, quality=10)


def create_batch_scene_graph(data, edge_radius, use_old_method=True):
    """
    Construct a spatiotemporal graph from agent positions in a dataset.

    returns: sg: An aggregate SceneGraph of the dataset.
    """
    nodes = [x for x in data.keys() if isinstance(x, STGNode)]
    N = len(nodes)
    total_timesteps = data['traj_lengths'].shape[0] if use_old_method else np.sum(data['traj_lengths'])
    position_cube = np.zeros((total_timesteps, N, 2))
    inactive_nodes = np.zeros((total_timesteps, N), dtype=np.int8)
    adj_cube = None
    if not use_old_method:
        adj_cube = np.zeros((data['traj_lengths'].shape[0], max(data['traj_lengths']), N, N), dtype=np.int8)

    for node_idx, node in enumerate(nodes):
        idx = 0
        for data_idx in range(data[node].shape[0]):
            if use_old_method:
                data_mat = data[node][data_idx, :data['traj_lengths'][data_idx], :2]
                position_cube[idx : idx + 1, node_idx] = data_mat[:1]
                inactive_nodes[idx : idx + 1, node_idx] = not data_mat[:1].any()
                idx += 1

            else:
                data_mat = data[node][data_idx, :data['traj_lengths'][data_idx], :2]
                position_cube[idx : idx + data['traj_lengths'][data_idx], node_idx] = data_mat
                inactive_nodes[idx : idx + data['traj_lengths'][data_idx], node_idx] = not data_mat.any()
                idx += data['traj_lengths'][data_idx]

    agg_adj_matrix = np.zeros((N, N), dtype=np.int8)
    if not use_old_method:
        curr_data_idx = 0
        curr_timestep = 0
        curr_sum = 0

    for timestep in range(position_cube.shape[0]):
        dists = squareform(pdist(position_cube[timestep], metric='euclidean'))

        # Put a 1 for all agent pairs which are closer than the edge_radius.
        adj_matrix = (dists <= edge_radius).astype(np.int8)

        # Remove self-loops.
        np.fill_diagonal(adj_matrix, 0)

        inactive_idxs = np.nonzero(inactive_nodes[timestep])

        adj_matrix[:, inactive_idxs] = 0
        adj_matrix[inactive_idxs, :] = 0

        agg_adj_matrix |= adj_matrix

        if not use_old_method:
            if timestep == (data['traj_lengths'][curr_data_idx] + curr_sum):
                curr_sum += data['traj_lengths'][curr_data_idx]
                curr_data_idx += 1
                curr_timestep = 0

            adj_cube[curr_data_idx, curr_timestep] = adj_matrix
            curr_timestep += 1

    sg = SceneGraph()
    sg.create_from_adj_matrix(agg_adj_matrix, nodes, edge_radius, adj_cube=adj_cube)

    return sg


def get_adj_matrix_helper(pos_matrix, edge_radius, nodes):
    N = len(nodes)
    active_idxs = list()

    dists = squareform(pdist(pos_matrix, metric='euclidean'))

    for idx in range(pos_matrix.shape[0]):
        if np.asarray(pos_matrix[idx]).any():
            active_idxs.append(idx)

    # Put a 1 for all agent pairs which are closer than the edge_radius.
    adj_matrix = (dists <= edge_radius).astype(int)
    assert len(adj_matrix.shape) == 2 and adj_matrix.shape == (N, N)

    # Remove self-loops.
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix, active_idxs


if __name__ == '__main__':
    ### Testing edge mask calculation ###
    A = np.array([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]])[:, :, np.newaxis, np.newaxis]
    print(A.shape)

    # (data_id, time, N, N)
    edge_addition_filter = [0.1, 0.2, 1.]
    edge_removal_filter = [1.,0.1]
    new_edges = np.minimum(ss.fftconvolve(A, np.reshape(edge_addition_filter, (1, -1, 1, 1)), 'full'), 1.)[:, (len(edge_addition_filter) - 1):]
    old_edges = np.minimum(ss.fftconvolve(A, np.reshape(edge_removal_filter, (1, -1, 1, 1)), 'full'), 1.)[:, :-(len(edge_removal_filter) - 1)]
    print(np.minimum(new_edges + old_edges, 1.))

    ############### Testing graph subtraction ###############
    # # # # # # # # # # # # # # #
    print('\n' + '-'*40 + '\n')

    scene_dict1 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (0, 1)}
    sg1 = SceneGraph()
    sg1.create_from_scene_dict(scene_dict1, edge_radius=5.)

    scene_dict2 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (1, 1)}
    sg2 = SceneGraph()
    sg2.create_from_scene_dict(scene_dict2, edge_radius=5.)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-'*40 + '\n')

    scene_dict1 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (0, 1)}
    sg1 = SceneGraph()
    sg1.create_from_scene_dict(scene_dict1, edge_radius=5.)

    scene_dict2 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (1, 1),
                   STGNode('3', 'Pedestrian'): (20, 1)}
    sg2 = SceneGraph()
    sg2.create_from_scene_dict(scene_dict2, edge_radius=5.)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-'*40 + '\n')

    scene_dict1 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (0, 1)}
    sg1 = SceneGraph()
    sg1.create_from_scene_dict(scene_dict1, edge_radius=5.)

    scene_dict2 = {STGNode('1', 'Pedestrian'): (1, 0),
                   STGNode('2', 'Pedestrian'): (10, 1),
                   STGNode('3', 'Pedestrian'): (20, 1)}
    sg2 = SceneGraph()
    sg2.create_from_scene_dict(scene_dict2, edge_radius=5.)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)

    # # # # # # # # # # # # # # #
    print('\n' + '-'*40 + '\n')

    scene_dict1 = {STGNode('1', 'Pedestrian'): (0, 0),
                   STGNode('2', 'Pedestrian'): (0, 1)}
    sg1 = SceneGraph()
    sg1.create_from_scene_dict(scene_dict1, edge_radius=5.)

    scene_dict2 = {STGNode('2', 'Pedestrian'): (10, 1),
                   STGNode('3', 'Pedestrian'): (12, 1)}
    sg2 = SceneGraph()
    sg2.create_from_scene_dict(scene_dict2, edge_radius=5.)

    new_nodes, removed_nodes, new_neighbors, removed_neighbors = sg2 - sg1
    print('New Nodes:', new_nodes)
    print('Removed Nodes:', removed_nodes)
    print('New Neighbors:', new_neighbors)
    print('Removed Neighbors:', removed_neighbors)
