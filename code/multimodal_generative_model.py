from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.contrib.distributions as distributions
import numpy as np

from stg_node import *
from utils.learning import *
from utils.learning import _SUPER_SECRET_EVAL_KEY
from utils.input_tensor_summarizer import *


class MultimodalGenerativeModel(object):
    def __init__(self, node, robot_node,
                 scene_graph,
                 instance_connected_to_robot,
                 edge_state_combine_method='sum',
                 edge_influence_combine_method='bi-rnn',
                 dynamic_edges='no'):
        
        self.node = node
        self.robot_node = robot_node
        self.scene_graph = scene_graph

        self.neighbors_via_edge_type = scene_graph.node_edges_and_neighbors[node]
        self.instance_connected_to_robot = instance_connected_to_robot
        self.edge_state_combine_method = edge_state_combine_method
        self.edge_influence_combine_method = edge_influence_combine_method
        self.dynamic_edges = dynamic_edges

    def setup_model(self, features, labels, mode, hps):
        pass

    def set_annealing_params(self):
        self.logging = {}
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("batchwise_annealing"):
                self.lr = exp_anneal(self.hps.learning_rate, self.hps.min_learning_rate, self.hps.learning_decay_rate)
                self.logging[str(self.node) + "/lr"] = self.lr
                tf.summary.scalar("lr", self.lr, family=self.node.name.replace(' ', '_'))

    def train_loss(self, tensor_dict):
        raise Exception("loss function must be overridden by child class of MultimodalGenerativeModel")

    def eval_loss(self, tensor_dict):
        return self.train_loss(tensor_dict)

    def standardize_features(self, features, mode):
        # features: {node1, node2, ..., traj_lengths}
        # feature_standardization: subdictionary of hps
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("features_standardization"):
                feature_standardization_vars = dict()
                nodes_standardization = self.hps.nodes_standardization
                if mode == tf.estimator.ModeKeys.TRAIN:
                    for node in nodes_standardization:
                        m = tf.get_variable(name="train_mean", 
                                            shape=nodes_standardization[node]["mean"].shape,
                                            initializer=tf.constant_initializer(nodes_standardization[node]["mean"]), 
                                            trainable=False)
                        s = tf.get_variable(name="train_std",
                                            shape=nodes_standardization[node]["std"].shape,
                                            initializer=tf.constant_initializer(nodes_standardization[node]["std"]),
                                            trainable=False)
                        feature_standardization_vars[node] = {"mean": m, "std": s}
                    
                    extras_standardization = self.hps.extras_standardization
                    em = tf.get_variable(name="train_extras_mean",
                                         shape=extras_standardization["mean"].shape,
                                         initializer=tf.constant_initializer(extras_standardization["mean"]), 
                                         trainable=False)
                    es = tf.get_variable(name="train_extras_std",
                                         shape=extras_standardization["std"].shape,
                                         initializer=tf.constant_initializer(extras_standardization["std"]),
                                         trainable=False)
                    
                else:    # these variables will get "restore"d over during EVAL/PREDICT
                    zero_state = np.zeros(self.state_dim, dtype=np.float32)
                    for node in nodes_standardization:
                        m = tf.get_variable(name="train_mean",
                                            shape=zero_state.shape,
                                            initializer=tf.constant_initializer(zero_state),
                                            trainable=False)
                        s = tf.get_variable(name="train_std",
                                            shape=zero_state.shape,
                                            initializer=tf.constant_initializer(zero_state),
                                            trainable=False)
                        feature_standardization_vars[node] = {"mean": m, "std": s}
                    
                    zero_extras = np.zeros(self.extras_dim, dtype=np.float32)
                    em = tf.get_variable(name="train_extras_mean",
                                         shape=zero_extras.shape,
                                         initializer=tf.constant_initializer(zero_extras),
                                         trainable=False)
                    es = tf.get_variable(name="train_extras_std",
                                         shape=zero_extras.shape,
                                         initializer=tf.constant_initializer(zero_extras),
                                         trainable=False)

                def standardize_feature(key, mean, std):
                    if mode == tf.estimator.ModeKeys.PREDICT or _SUPER_SECRET_EVAL_KEY in features:
                        key = str(key)
                                        
                    with tf.variable_scope(str(key)):
                        std_data = standardize(tf.to_float(features[key]), mean, std)
                        if mode == tf.estimator.ModeKeys.TRAIN and self.hps.fuzz_factor > 0:
                            return std_data + self.hps.fuzz_factor*tf.random_normal(std_data.shape)
                        return std_data

                features_standardized = {
                    "extras": standardize_feature("extras", em, es),        # [batch_size, max_time, extras_dim]
                    "traj_lengths": tf.to_int32(features["traj_lengths"])   # [batch_size]
                }
                
                for node in nodes_standardization:
                    features_standardized[node] = standardize_feature(node, 
                                                                      feature_standardization_vars[node]["mean"],
                                                                      feature_standardization_vars[node]["std"]) 
                                                  # [batch_size, max_time, state_dim]
                
                if "bag_idx" in features:
                    features_standardized["bag_idx"] = tf.to_int32(features["bag_idx"])  # [batch_size, max_time, 1]

                if "edge_scaling_mask" in features:
                    features_standardized["edge_scaling_mask"] = tf.to_float(features["edge_scaling_mask"])  # [batch_size, max_time, N, N]

                if mode == tf.estimator.ModeKeys.PREDICT:
                    robot_future = str(self.robot_node) + "_future"
                    features_standardized[robot_future] = standardize_feature(robot_future, m, s)

            return features_standardized

    def standardize_labels(self, labels, mode):
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("label_standardization"):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    labels_standardization = self.hps.labels_standardization
                    
                    label_node = convert_to_label_node(self.node)
                    m = tf.get_variable(name="train_mean", 
                                        shape=labels_standardization[label_node]["mean"].shape,
                                        initializer=tf.constant_initializer(labels_standardization[label_node]["mean"]), 
                                        trainable=False)
                    s = tf.get_variable(name="train_std",
                                        shape=labels_standardization[label_node]["std"].shape,
                                        initializer=tf.constant_initializer(labels_standardization[label_node]["std"]),
                                        trainable=False)
                    
                else:    # these variables will get "restore"d over during EVAL/PREDICT
                    zero_state = np.zeros(self.pred_dim, dtype=np.float32)
                    
                    m = tf.get_variable(name="train_mean",
                                        shape=zero_state.shape,
                                        initializer=tf.constant_initializer(zero_state),
                                        trainable=False)
                    
                    s = tf.get_variable(name="train_std",
                                        shape=zero_state.shape,
                                        initializer=tf.constant_initializer(zero_state),
                                        trainable=False)

                if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                    std_data = standardize(tf.to_float(labels), m, s)
                    if mode == tf.estimator.ModeKeys.TRAIN and self.hps.fuzz_factor > 0:
                        return std_data + self.hps.fuzz_factor*tf.random_normal(std_data.shape)
                    return std_data
                
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    self.labels_m = m
                    self.labels_s = s
                    return None

    # NOT MEANT TO BE USED BY ESTIMATOR
    # This is simply called so that each node manages its own loss and etc.
    def model_fn(self, features, labels, mode, params):
        self.setup_model(features, labels, mode, params)

        self.mode = mode
        self.hps = params
        self.predictions_dict = None
        self.eval_metric_ops = None
        self.eval_ops = None
        self.loss = None
        self.train_op = None
        self.temp = None
        self.lr = None
        self.kl_weight = None
        self.logging = {}

        # standardize the features, returns a dictionary of {car1, car2, traj_lengths}
        features_standardized = self.standardize_features(features, mode)
        # standardize the labels
        labels_standardized = self.standardize_labels(labels, mode)

        # prepares the features by using RNNs to summarize
        tensor_dict = InputTensorSummarizer(features_standardized, labels_standardized, 
                                            mode, params, 
                                            self.node, self.robot_node, 
                                            self.scene_graph,
                                            self.instance_connected_to_robot,
                                            self.edge_state_combine_method,
                                            self.edge_influence_combine_method,
                                            self.dynamic_edges).tensor_dict

        if mode == tf.estimator.ModeKeys.TRAIN:
            # annealing function <- inputs into other functions?
            self.set_annealing_params()
            # get loss function <- where all the subclasses get real
            self.loss = self.train_loss(tensor_dict)
        elif mode == tf.estimator.ModeKeys.EVAL:
            self.loss = self.eval_loss(tensor_dict)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            self.predictions_dict = self.make_predictions_dict(tensor_dict)


class MultimodalGenerativeCVAE(MultimodalGenerativeModel):
    
    def setup_model(self, features, labels, mode, hps):
        if mode == tf.estimator.ModeKeys.PREDICT or _SUPER_SECRET_EVAL_KEY in features:
            naming_func = lambda x: str(x)
        else:
            naming_func = lambda x: x
        
        robot_future_x = str(self.robot_node) + "_future_x"
        robot_future_y = str(self.robot_node) + "_future_y"
        robot_future = str(self.robot_node) + "_future"
        
        if robot_future_x in features and robot_future_y in features:
            features[robot_future] = cartesian_product_over_batch(features[robot_future_x],
                                                                   features[robot_future_y],
                                                                   name=robot_future)
            features.pop(robot_future_x)
            features.pop(robot_future_y)
        
        if isinstance(hps.pred_indices, dict):
            self.pred_dim = len(hps.pred_indices[self.node.type])    # labels.shape[-1].value
        else:
            self.pred_dim = len(hps.pred_indices)    # labels.shape[-1].value
        self.state_dim = features[naming_func(self.robot_node)].shape[-1].value
        self.extras_dim = features["extras"].shape[-1].value
        
        if hps.latent_type == "MVG":
            self.latent = MVGLatent(hps.MVG_latent_dim, self.node)
        elif hps.latent_type == "discrete":
            # N = number of variables, K = categories per variable
            self.latent = DiscreteLatent(hps.N, hps.K, self.node)
            self.latent.kl_min = hps.kl_min

        with tf.variable_scope("sample_ct"):
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.sample_ct = hps.k
            elif mode == tf.estimator.ModeKeys.EVAL:
                self.sample_ct = hps.k_eval
            elif mode == tf.estimator.ModeKeys.PREDICT:
                self.sample_ct = features["sample_ct"][0]

    def set_annealing_params(self):
        super(MultimodalGenerativeCVAE, self).set_annealing_params()
        
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("batchwise_annealing", reuse=True):
                if np.abs(self.hps.alpha - 1.0) < 1e-3 and not self.hps.use_iwae:
                    self.kl_weight = sigmoid_anneal(self.hps.kl_weight_start, self.hps.kl_weight,
                                                    self.hps.kl_crossover, self.hps.kl_crossover / self.hps.kl_sigmoid_divisor)
                    self.logging[str(self.node) + "/kl_weight"] = self.kl_weight
                    tf.summary.scalar("kl_weight", self.kl_weight, family=self.node.name.replace(' ', '_'))

                if self.hps.sample_model_during_dec:
                    self.dec_sample_model_prob = sigmoid_anneal(
                        self.hps.dec_sample_model_prob_start,
                        self.hps.dec_sample_model_prob_final,
                        self.hps.dec_sample_model_prob_crossover,
                        self.hps.dec_sample_model_prob_crossover / self.hps.dec_sample_model_prob_divisor
                    )
                    tf.summary.scalar("dec_sample_model_prob", self.dec_sample_model_prob, family=self.node.name.replace(' ', '_'))

                if self.hps.latent_type == "discrete":
                    self.latent.temp = exp_anneal(self.hps.tau_init, self.hps.tau_final, self.hps.tau_decay_rate)
                    self.logging[str(self.node) + "/temp"] = self.latent.temp
                    tf.summary.scalar("temp", self.latent.temp, family=self.node.name.replace(' ', '_'))
                    if self.hps.use_z_logit_clipping:
                        self.latent.z_logit_clip = sigmoid_anneal(self.hps.z_logit_clip_start, self.hps.z_logit_clip_final,
                                                                  self.hps.z_logit_clip_crossover,
                                                                  self.hps.z_logit_clip_crossover / self.hps.z_logit_clip_divisor)
                        tf.summary.scalar("z_logit_clip", self.z_logit_clip, family=self.node.name.replace(' ', '_'))

    def q_z_xy(self, x, y, mode):
        with tf.variable_scope("q_z_xy"):
            xy = tf.concat([x, y], 1)
            # h = xy    # https://arxiv.org/pdf/1703.10960.pdf, https://arxiv.org/pdf/1704.03477.pdf
            h = MLP(xy, self.hps.q_z_xy_MLP_dims, tf.nn.relu, self.hps.MLP_dropout_keep_prob, mode)
            return self.latent.dist_from_h(h, mode)

    def p_z_x(self, x, mode):
        with tf.variable_scope("p_z_x"):
            # h = tf.layers.dense(x, h_dim, activation=tf.nn.relu, name="dense")    # https://arxiv.org/pdf/1703.10960.pdf
            h = MLP(x, self.hps.p_z_x_MLP_dims, tf.nn.relu, self.hps.MLP_dropout_keep_prob, mode)
            return self.latent.dist_from_h(h, mode)

    def p_y_xz(self, x, z_stacked, TD, mode):
        # x is [bs/nbs, 2*enc_rnn_dim]
        # z_stacked is [k, bs/nbs, N*K]    (at EVAL or PREDICT time, k (=self.sample_ct) may be hps.k, K**N or sample_ct)
        # in this function, rnn decoder inputs are of the form: z + x + car1 + car2 (note: first 3 are "extras" to help with learning)
        ph = self.hps.prediction_horizon
        
        robot_future = str(self.robot_node) + "_future"
        our_future = str(self.node) + "_future"

        k, GMM_c, pred_dim = self.sample_ct, self.hps.GMM_components, self.pred_dim
        with tf.variable_scope("p_y_xz"):
            z = tf.reshape(z_stacked, [-1, self.latent.z_dim])               # [k;bs/nbs, z_dim]
            zx = tf.concat([z, tf.tile(x, [k, 1])], axis=1)           # [k;bs/nbs, z_dim + 2*enc_rnn_dim]

            cell = stacked_rnn_cell(self.hps.rnn_cell,
                                    self.hps.rnn_cell_kwargs,
                                    self.hps.dec_rnn_dim,
                                    self.hps.rnn_io_dropout_keep_prob,
                                    mode)
            initial_state = project_to_RNN_initial_state(cell, zx)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                if self.hps.sample_model_during_dec and mode == tf.estimator.ModeKeys.TRAIN:
                    input_ = tf.concat([zx, tf.tile(TD["joint_present"], [k, 1])], axis=1)    # [k;bs, N*K + 2*enc_rnn_dim + pred_dim+state_dim]
                    state = initial_state
                    with tf.variable_scope("rnn") as rnnscope:
                        log_pis, mus, log_sigmas, corrs = [], [], [], []
                        for j in range(ph):
                            if j > 0:
                                rnnscope.reuse_variables()
                            output, state = cell(input_, state)
                            log_pi_t, mu_t, log_sigma_t, corr_t = project_to_GMM_params(output, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims) 
                            y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                        self.hps.log_sigma_min, self.hps.log_sigma_max).sample()              # [k;bs, pred_dim]
                            mask = distributions.Bernoulli(probs=self.dec_sample_model_prob,
                                                           dtype=tf.float32).sample((tf.shape(y_t)[0], 1))    # maybe tf.shape
                            y_t = mask*y_t + (1 - mask)*tf.tile(TD[our_future][:,j,:], [k, 1])
                            log_pis.append(log_pi_t)
                            mus.append(mu_t)
                            log_sigmas.append(log_sigma_t)
                            corrs.append(corr_t)
                            car_inputs = tf.concat([tf.tile(TD[robot_future][:,j,:], [k, 1]), y_t], axis=1)  # [k;bs, state_dim + pred_dim]
                            input_ = tf.concat([zx, car_inputs], axis=1)                # [k;bs, N*K + 2*enc_rnn_dim + state_dim + pred_dim]  
                        log_pis = tf.stack(log_pis, axis=1)                             # [k;bs, ph, GMM_c]
                        mus = tf.stack(mus, axis=1)                                     # [k;bs, ph, GMM_c*pred_dim]
                        log_sigmas = tf.stack(log_sigmas, axis=1)                       # [k;bs, ph, GMM_c*pred_dim]
                        corrs = tf.stack(corrs, axis=1)                                 # [k;bs, ph, GMM_c]
                else:
                    zx_with_time_dim = tf.expand_dims(zx, 1)                           # [k;bs/nbs, 1, N*K + 2*enc_rnn_dim]
                    zx_time_tiled = tf.tile(zx_with_time_dim, [1, ph, 1])              # [k;bs/nbs, ph, N*K + 2*enc_rnn_dim]
                    car_inputs = tf.concat([                                           # [bs/nbs, ph, 2*state_dim]
                        tf.expand_dims(TD["joint_present"], 1),                                          # [bs/nbs, 1, state_dim+pred_dim]
                        tf.concat([TD[robot_future][:,:ph-1,:], TD[our_future][:,:ph-1,:]], axis=2)  # [bs/nbs, ph-1, state_dim+pred_dim]
                        ], axis=1)
                    inputs = tf.concat([zx_time_tiled, tf.tile(car_inputs, [k, 1, 1])], axis=2)  # [k;bs/nbs, ph, N*K + 2*enc_rnn_dim + pred_dim + state_dim]
                    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state,    # [k;bs/nbs, ph, dec_rnn_dim]
                                                                 time_major=False,
                                                                 dtype=tf.float32,
                                                                 scope="rnn")
                    with tf.variable_scope("rnn"):    # required to match PREDICT mode below
                        log_pis, mus, log_sigmas, corrs = project_to_GMM_params(outputs, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims)

                tf.summary.histogram("GMM_log_pis", log_pis, family=self.node.name.replace(' ', '_'))
                tf.summary.histogram("GMM_mus", mus, family=self.node.name.replace(' ', '_'))
                tf.summary.histogram("GMM_log_sigmas", log_sigmas, family=self.node.name.replace(' ', '_'))
                tf.summary.histogram("GMM_corrs", corrs, family=self.node.name.replace(' ', '_'))

            elif mode == tf.estimator.ModeKeys.PREDICT:
                input_ = tf.concat([zx, tf.tile(TD["joint_present"], [k, 1])], axis=1)    # [k;bs, N*K + 2*enc_rnn_dim + pred_dim+state_dim]
                state = initial_state
                with tf.variable_scope("rnn") as rnnscope:
                    log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
                    for j in range(ph):
                        if j > 0:
                            rnnscope.reuse_variables()
                        output, state = cell(input_, state)
                        log_pi_t, mu_t, log_sigma_t, corr_t = project_to_GMM_params(output, GMM_c, pred_dim, self.hps.dec_GMM_proj_MLP_dims) 
                        y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t,
                                    self.hps.log_sigma_min, self.hps.log_sigma_max).sample()              # [k;bs, pred_dim]
                        log_pis.append(log_pi_t)
                        mus.append(mu_t)
                        log_sigmas.append(log_sigma_t)
                        corrs.append(corr_t)
                        y.append(y_t)
                        car_inputs = tf.concat([tf.tile(TD[robot_future][:,j,:], [k, 1]), y_t], axis=1)  # [k;bs, state_dim + pred_dim]
                        input_ = tf.concat([zx, car_inputs], axis=1)                # [k;bs, N*K + 2*enc_rnn_dim + state_dim + pred_dim]  
                    log_pis = tf.stack(log_pis, axis=1)                             # [k;bs, ph, GMM_c]
                    mus = tf.stack(mus, axis=1)                                     # [k;bs, ph, GMM_c*pred_dim]
                    log_sigmas = tf.stack(log_sigmas, axis=1)                       # [k;bs, ph, GMM_c*pred_dim]
                    corrs = tf.stack(corrs, axis=1)                                 # [k;bs, ph, GMM_c]
                    car2_sampled_future = tf.reshape(tf.stack(y, axis=1), [k, -1, ph, pred_dim])  # [k, bs, ph, pred_dim]

            y_dist = GMM2D(tf.reshape(log_pis, [k, -1, ph, GMM_c]),
                           tf.reshape(mus, [k, -1, ph, GMM_c*pred_dim]),
                           tf.reshape(log_sigmas, [k, -1, ph, GMM_c*pred_dim]),
                           tf.reshape(corrs, [k, -1, ph, GMM_c]),
                           self.hps.log_sigma_min,
                           self.hps.log_sigma_max)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return y_dist, car2_sampled_future
            else:
                return y_dist

    def encoder(self, x, y, mode):
        k = self.sample_ct
        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("encoder"):
                self.latent.q_dist = self.q_z_xy(x, y, mode)
                self.latent.p_dist = self.p_z_x(x, mode)
                z = self.latent.sample_q(k, mode)
                if mode == tf.estimator.ModeKeys.TRAIN and self.hps.kl_exact:
                    kl_obj = self.latent.kl_q_p()
                    tf.summary.scalar("kl", kl_obj, family=self.node.name.replace(' ', '_'))
                else:
                    kl_obj = None

                return z, kl_obj

    def decoder(self, x, y, z, TD, mode):
        # x is [nbs, 2*enc_rnn_dim]
        # y is [nbs, prediction_horizon, pred_dim]
        # z is [k, bs/nbs, N*K]
        ph = self.hps.prediction_horizon
        pred_dim = self.pred_dim

        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("decoder"):
                self.y_dist = y_dist = self.p_y_xz(x, z, TD, mode)
                if len(y_dist.batch_shape) == 2:    # e.g., y_dist is a MVG with loc (mean) shape [k, nbs, ph*pred_dim]
                    y_vector = tf.reshape(y, [-1, ph*pred_dim])               # [nbs, ph*pred_dim]
                    log_p_y_xz = y_dist.log_prob(y_vector)                    # [k, nbs]
                elif len(y_dist.batch_shape) == 3:  # e.g., y_dist is a GMM with mus [k, nbs, ph, GMM_c;pred_dim]
                    log_p_yt_xz = tf.minimum(y_dist.log_prob(y), self.hps.log_p_yt_xz_max)
                    log_p_y_xz = tf.reduce_sum(log_p_yt_xz, axis=2)    # [k, nbs]
                    tf.summary.histogram("log_p_yt_xz", log_p_yt_xz, family=self.node.name.replace(' ', '_'))
                return log_p_y_xz

    def train_loss(self, TD):
        mode = tf.estimator.ModeKeys.TRAIN
        z, kl = self.encoder(TD["x"], TD[self.node.type + "_future_encoder"], mode)        # [k, nbs, N*K], [] (i.e., scalar)
        log_p_y_xz = self.decoder(TD["x"], TD[str(self.node) + "_future"], z, TD, mode)          # [k, nbs]

        if np.abs(self.hps.alpha - 1.0) < 1e-3 and not self.hps.use_iwae:
            log_p_y_xz_mean = tf.reduce_mean(log_p_y_xz, 0)                       # [nbs]
            tf.summary.histogram("log_p_y_xz", log_p_y_xz_mean, family=self.node.name.replace(' ', '_'))
            log_likelihood = tf.reduce_mean(log_p_y_xz_mean)
            ELBO = log_likelihood - self.kl_weight*kl
            loss = -ELBO
        else:
            log_q_z_xy = self.latent.q_log_prob(z)    # [k, nbs]
            log_p_z_x = self.latent.p_log_prob(z)     # [k, nbs]
            a = self.hps.alpha
            log_pp_over_q = log_p_y_xz + log_p_z_x - log_q_z_xy
            log_likelihood = (tf.reduce_mean(tf.reduce_logsumexp(log_pp_over_q*(1-a), axis=0)) -\
                                   tf.log(tf.to_float(self.hps.k))) / (1-a)
            loss = -log_likelihood

        tf.summary.scalar("log_likelihood", log_likelihood, family=self.node.name.replace(' ', '_'))
        tf.summary.scalar("loss", loss, family=self.node.name.replace(' ', '_'))
        self.latent.summarize_for_tensorboard()
        return loss

    def eval_loss(self, TD):
        mode = tf.estimator.ModeKeys.EVAL
        our_future = str(self.node) + "_future"

        ### Importance sampled NLL estimate
        z, _ = self.encoder(TD["x"], TD[self.node.type + "_future_encoder"], mode)      # [k_eval, nbs, N*K]
        log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode)       # [k_eval, nbs]
        log_q_z_xy = self.latent.q_log_prob(z)                          # [k_eval, nbs]
        log_p_z_x = self.latent.p_log_prob(z)                           # [k_eval, nbs]
        log_likelihood = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz + log_p_z_x - log_q_z_xy, axis=0)) -\
                         tf.log(tf.to_float(self.sample_ct))
        loss = -log_likelihood
        
        nll_q_name = str(self.node) + "/NLL_q_IS"
        nll_q_is = tf.identity(-log_likelihood, name=nll_q_name)
        self.eval_ops = {nll_q_name: nll_q_is}
        self.eval_metric_ops = {nll_q_name: tf.metrics.mean(nll_q_is)}

        ### Naive sampled NLL estimate
        z = self.latent.sample_p(self.sample_ct, mode)
        log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode)
        log_likelihood_p = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz, axis=0)) - tf.log(tf.to_float(self.sample_ct))
        
        nll_p_name = str(self.node) + "/NLL_p"
        nll_p = tf.identity(-log_likelihood_p, name=nll_p_name)
        self.eval_ops[nll_p_name] = nll_p
        self.eval_metric_ops[nll_p_name] = tf.metrics.mean(nll_p)

        ### Exact NLL
        K, N = self.hps.K, self.hps.N
        if self.hps.latent_type == "discrete" and K**N < 50:
            self.sample_ct = K ** N
            nbs = tf.shape(TD["x"])[0]
            z_raw = tf.tile(all_one_hot_combinations(N, K, np.float32), multiples=[1, nbs])    # [K**N, nbs*N*K]
            z = tf.reshape(z_raw, [K**N, -1, N*K])                                                  # [K**N, nbs, N*K]
            log_p_y_xz = self.decoder(TD["x"], TD[our_future], z, TD, mode)                    # [K**N, nbs]
            log_p_z_x = self.latent.p_log_prob(z)                                              # [K**N, nbs]
            exact_log_likelihood = tf.reduce_mean(tf.reduce_logsumexp(log_p_y_xz + log_p_z_x, axis=0))
            
            nll_exact_name = str(self.node) + "/NLL_exact"
            nll_exact = tf.identity(-exact_log_likelihood, name=nll_exact_name)
            self.eval_ops[nll_exact_name] = nll_exact
            self.eval_metric_ops[nll_exact_name] = tf.metrics.mean(nll_exact)

        return loss

    def make_predictions_dict(self, TD):
        mode = tf.estimator.ModeKeys.PREDICT

        with tf.variable_scope(self.node.type, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("encoder"):
                self.latent.p_dist = self.p_z_x(TD["x"], mode)

            z = self.latent.sample_p(self.sample_ct, mode)

            with tf.variable_scope("decoder"):
                y_dist, our_sampled_future = self.p_y_xz(TD["x"], z, TD, mode)      # y_dist.mean is [k, bs, ph*state_dim]

        with tf.variable_scope(str(self.node)):
            with tf.variable_scope("outputs"):
                y = unstandardize(our_sampled_future, self.labels_m, self.labels_s)

                predictions_dict = {"y": y, "z": z}
                predictions_dict = {k: tf.identity(v, name=k) for k,v in predictions_dict.items()}
                return predictions_dict
