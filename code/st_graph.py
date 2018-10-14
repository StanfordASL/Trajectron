from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

from stg_node import *
from utils.learning import *
from utils.learning import _SUPER_SECRET_EVAL_KEY
from multimodal_generative_model import *


hps = tf.contrib.training.HParams(
    ### Logging
    steps_per_log = 10,
    
    ### Training
    ## Batch Sizes
    batch_size = 16,
    ## Learning Rate
    learning_rate = 0.001,
    min_learning_rate = 0.00001,
    learning_decay_rate = 0.9999,
    ## Optimizer
    optimizer = tf.train.AdamOptimizer,
    optimizer_kwargs = {},
    grad_clip = 1.0,

    ### Prediction
    minimum_history_length = 5,    # 0.5 seconds
    prediction_horizon = 15,       # 1.5 seconds (at least as far as the loss function is concerned)

    ### Variational Objective
    ## Objective Formulation
    alpha = 1,
    k = 3,              # number of samples from z during training
    k_eval = 50,        # number of samples from z during evaluation
    use_iwae = False,   # only matters if alpha = 1
    kl_exact = True,    # relevant only if alpha = 1
    ## KL Annealing/Bounding
    kl_min = 0.07,
    kl_weight = 1.0,
    kl_weight_start = 0.0001,
    kl_decay_rate = 0.99995,
    kl_crossover = 8000,
    kl_sigmoid_divisor = 6,

    ### Network Parameters
    ## RNNs/Summarization
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell,
    rnn_cell_kwargs = {"layer_norm": False, "dropout_keep_prob": 0.75},
    MLP_dropout_keep_prob = 0.9,
    rnn_io_dropout_keep_prob = 1.0,
    enc_rnn_dim_multiple_inputs = [8],
    enc_rnn_dim_edge = [8],
    enc_rnn_dim_edge_influence = [8],
    enc_rnn_dim_history = [32],
    enc_rnn_dim_future = [32],
    dec_rnn_dim = [128],
    dec_GMM_proj_MLP_dims = None,
    sample_model_during_dec = True,
    dec_sample_model_prob_start = 0.0,
    dec_sample_model_prob_final = 0.0,
    dec_sample_model_prob_crossover = 20000,
    dec_sample_model_prob_divisor = 6,
    ## q_z_xy (encoder)
    q_z_xy_MLP_dims = None,
    ## p_z_x (encoder)
    p_z_x_MLP_dims = [16],
    ## p_y_xz (decoder)
    fuzz_factor = 0.05,
    GMM_components = 16,
    log_sigma_min = -10,
    log_sigma_max = 10,
    log_p_yt_xz_max = 50,

    ### Latent Variables
    latent_type = "discrete",
    ## Discrete Latent
    N = 2,
    K = 5,
    # Relaxed One-Hot Temperature Annealing
    tau_init = 2.0,
    tau_final = 0.001,
    tau_decay_rate = 0.9999,
    # Logit Clipping
    use_z_logit_clipping = False,
    z_logit_clip_start = 0.05,
    z_logit_clip_final = 3.0,
    z_logit_clip_crossover = 8000,
    z_logit_clip_divisor = 6,
    ## MVG Latent
    MVG_latent_dim = 32
)


class SpatioTemporalGraphCVAE(object):
    def __init__(self, scene_graph, robot_node,
                 edge_radius=5.0, 
                 edge_state_combine_method='sum',
                 edge_influence_combine_method='bi-rnn',
                 dynamic_edges='no'):
        
        self.robot_node = robot_node
        self.edge_radius = edge_radius
        
        self.scene_graph = scene_graph
        self.all_nodes = scene_graph.nodes
        self.edge_types = scene_graph.edge_types
        self.node_edges_and_neighbors = scene_graph.node_edges_and_neighbors
        self.dynamic_edges = dynamic_edges
                    
        N = len(self.all_nodes)

        # Determining which agents are connected to the robot.
        instance_connected_to_robot = {node: False for node in self.all_nodes}
        for node in self.all_nodes:
            for edge_type in self.edge_types[node]:
                if robot_node.type in edge_type:
                    instance_connected_to_robot[node] = True
                    break
        
        self.node_model_dict = dict()
        for node in self.all_nodes:
            if node != robot_node:
                self.node_model_dict[node] = MultimodalGenerativeCVAE(node, robot_node, scene_graph, 
                                                                      instance_connected_to_robot[node], 
                                                                      edge_state_combine_method=edge_state_combine_method,
                                                                      edge_influence_combine_method=edge_influence_combine_method,
                                                                      dynamic_edges=dynamic_edges)
    
    
    def get_adj_matrix(self, pos_matrix):
        N = pos_matrix.shape[0]
        
        dists = squareform(pdist(pos_matrix, metric='euclidean'))

        # Put a 1 for all agent pairs which are closer than the edge_radius.
        adj_matrix = (dists <= self.edge_radius).astype(int)
        
        assert len(adj_matrix.shape) == 2 and adj_matrix.shape == (N, N)

        # Remove self-loops.
        np.fill_diagonal(adj_matrix, 0)
        
        return adj_matrix
        
    
    def get_st_graph_info(self, pos_dict):
        """
        Construct a spatiotemporal graph from N agent positions.

        pos_dict: nbags x N x 2 dict describing the initial x and y position of each agent per bag.

        returns: node_names: An N-length list of ordered node name strings.
                 edge_types: An N-size dict containing lists of edge-type string names per node.
                 all_node_edges_and_neighbors: 
        """

        all_nodes = set()
        all_edge_types = defaultdict(set)
        all_node_edges_and_neighbors = dict()
        
        for bagname in pos_dict:
            node_type_pos_dict = pos_dict[bagname]
            N = len(node_type_pos_dict)
                        
            nodes = list()
            for (node_name, node_type) in node_type_pos_dict.keys():
                nodes.append(STGNode(node_name, node_type))
            assert len(nodes) == N
            
            pos_matrix = np.array([list(node_type_pos_dict[node_type_pair]) for node_type_pair in node_type_pos_dict])
            assert pos_matrix.shape == (N, 2)

            adj_matrix = self.get_adj_matrix(pos_matrix)
            assert adj_matrix.shape == (N, N)
            
            node_edges_and_neighbors = {node: defaultdict(set) for node in nodes}
            edge_types = defaultdict(list)
            for i in xrange(N):
                curr_node = nodes[i]
                for j in xrange(N):
                    curr_neighbor = nodes[j]
                    if adj_matrix[i, j] == 1:
                        sorted_edge_type = sorted([curr_node.type, curr_neighbor.type])
                        edge_type = '-'.join(sorted_edge_type)
                        edge_types[curr_node].append(edge_type)

                        node_edges_and_neighbors[curr_node][edge_type].add(curr_neighbor)

            all_nodes.update(nodes)
                        
            for node in edge_types:
                all_edge_types[node].update(edge_types[node])
                
            for node in node_edges_and_neighbors:
                if node in all_node_edges_and_neighbors:
                    for edge_type in node_edges_and_neighbors[node]:
                        if edge_type in all_node_edges_and_neighbors[node]:
                            all_node_edges_and_neighbors[node][edge_type].update(node_edges_and_neighbors[node][edge_type])
                        else:
                            all_node_edges_and_neighbors[node][edge_type] = node_edges_and_neighbors[node][edge_type]
                else:
                    all_node_edges_and_neighbors[node] = node_edges_and_neighbors[node]                
        
        # List-ifying these so looping over them later on is faster.
        for node in all_edge_types:
            all_edge_types[node] = list(all_edge_types[node])
            
        for node in all_node_edges_and_neighbors:
            for edge_type in all_node_edges_and_neighbors[node]:
                all_node_edges_and_neighbors[node][edge_type] = list(all_node_edges_and_neighbors[node][edge_type])
        
        return list(all_nodes), all_edge_types, all_node_edges_and_neighbors
        
    
    def set_annealing_params(self):
        self.logging = {}
        with tf.variable_scope("batchwise_annealing"):
            self.lr = exp_anneal(self.hps.learning_rate, self.hps.min_learning_rate, self.hps.learning_decay_rate)
            self.logging["lr"] = self.lr
            tf.summary.scalar("lr", self.lr, family='dynstg')
            
    
    def optimizer(self, loss):
        with tf.variable_scope("optimizer"):
            opt = self.hps.optimizer(learning_rate=self.lr, **self.hps.optimizer_kwargs)
            if self.hps.grad_clip is not None:
                gvs = opt.compute_gradients(self.loss)
                g = self.hps.grad_clip
                clipped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
                train_op = opt.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            else:
                train_op = opt.minimize(loss, tf.train.get_global_step())
        return train_op
    
    
    def model_fn(self, features, labels, mode, params):
        # This is a hack to export an eval model with export_savedmodel.
        naming_func = lambda x: x
        if _SUPER_SECRET_EVAL_KEY in features:
            mode = tf.estimator.ModeKeys.EVAL
                        
            labels = {k: v for k, v in features.iteritems() if k.endswith('_label')}
            for key in labels:
                features.pop(key)
            
            naming_func = lambda x: str(x)
        
        # Creating all the subgraph models.
        for i, (node, model) in enumerate(self.node_model_dict.iteritems()):
            if mode != tf.estimator.ModeKeys.PREDICT:
                labels_passed_in = labels[naming_func(convert_to_label_node(node))]
            else:
                labels_passed_in = labels
            
            model.model_fn(features, labels_passed_in, mode, params)

        self.mode = mode
        self.hps = params
        self.predictions_dict = None
        self.eval_metric_ops = None
        self.eval_ops = None
        self.loss = None
        self.train_op = None
        self.logging = dict()
        
        models = [value for value in self.node_model_dict.itervalues()]
        if mode == tf.estimator.ModeKeys.TRAIN:
            # annealing function <- inputs into other functions?
            self.set_annealing_params()
            
            # aggregate underlying training logging
            for model in models:
                self.logging.update(model.logging)
            
            # # sum losses from node models <- where all the subclasses get real
            self.loss = sum([model.loss for model in models])
            
            # optimizer
            self.train_op = self.optimizer(self.loss)
        
        elif mode == tf.estimator.ModeKeys.EVAL:
            # sum losses from node models
            self.loss = sum([model.loss for model in models])
            
            # aggregate their eval metric ops into one
            self.eval_metric_ops = dict()
            self.eval_ops = dict()
            for model in models:
                self.eval_metric_ops.update(model.eval_metric_ops)
                self.eval_ops.update(model.eval_ops)
                
            ### NLL_Q (IS)
            nll_q_is_values = [op for key, op in self.eval_ops.iteritems() if key.endswith('NLL_q_IS')]
            nll_q_is_name = 'ST-Graph/NLL_q_IS'
            nll_q_is = tf.identity(tf.reduce_mean(tf.stack(nll_q_is_values), axis=0), 
                                   name=nll_q_is_name)
            self.eval_metric_ops["NLL_q (IS)"] = tf.metrics.mean(nll_q_is)
            self.eval_ops[nll_q_is_name] = nll_q_is
            
            ### NLL_P
            nll_p_values = [op for key, op in self.eval_ops.iteritems() if key.endswith('NLL_p')]
            nll_p_name = 'ST-Graph/NLL_p'
            nll_p = tf.identity(tf.reduce_mean(tf.stack(nll_p_values), axis=0), 
                                name=nll_p_name)
            self.eval_metric_ops["NLL_p"] = tf.metrics.mean(nll_p)
            self.eval_ops[nll_p_name] = nll_p
            
            ### NLL_Exact
            nll_exact_values = [op for key, op in self.eval_ops.iteritems() if key.endswith('NLL_exact')]
            stacked_nll_exact_values = tf.stack(nll_exact_values)
            stacked_nll_exact_values = tf.boolean_mask(stacked_nll_exact_values, tf.is_finite(stacked_nll_exact_values))
            # stacked_nll_exact_values = tf.Print(stacked_nll_exact_values, [stacked_nll_exact_values], summarize=100, message='stacked_nll_exact_values')
            nll_exact_name = 'ST-Graph/NLL_exact'
            nll_exact = tf.identity(tf.reduce_mean(stacked_nll_exact_values, axis=0), 
                                    name=nll_exact_name)
            self.eval_metric_ops["NLL_exact"] = tf.metrics.mean(nll_exact)
            self.eval_ops[nll_exact_name] = nll_exact
                        
        elif mode == tf.estimator.ModeKeys.PREDICT:
            # Combine the nodes' prediction dicts.
            self.predictions_dict = dict()
            for model in models:
                self.predictions_dict.update(model.predictions_dict)
        
        if _SUPER_SECRET_EVAL_KEY in features:
            self.predictions_dict = self.eval_ops
        
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(self.predictions_dict)} if self.predictions_dict else None
        return tf.estimator.EstimatorSpec(mode, 
                                          self.predictions_dict, 
                                          self.loss, 
                                          self.train_op,
                                          # Could add a tf.train.ProfilerHook to profile model memory and compute information,
                                          # just need to figure out where to save the resulting output files (i.e. how to pass
                                          # in the model checkpoints directory, that'd be all I need to put this in).
                                          training_hooks=[tf.train.LoggingTensorHook(self.logging,
                                                                                     every_n_iter=self.hps.steps_per_log)],
                                          eval_metric_ops=self.eval_metric_ops,
                                          export_outputs=export_outputs)
