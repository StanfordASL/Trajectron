from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as distributions

from utils.bags import *


_SUPER_SECRET_EVAL_KEY = '_SUPER_SECRET_EVAL_KEY'


### DATA PREPROCESSING
def prepare_data(bag_df, extract_dict = full_extract_dict, pred_targets = ["xdd2", "ydd2"]):

    def extract_mean_and_std(A, tl, off=0):
        data = np.concatenate([A[i,:l+off,:] for i, l in enumerate(tl)], axis=0)
        return data.mean(axis=0).astype(np.float32), data.std(axis=0).astype(np.float32)

    car2states = extract_dict["car2"]["/object_state/2"].keys()
    car2state_to_index = dict(zip(car2states, range(len(car2states))))
    pred_indices = [car2state_to_index[t] for t in pred_targets]

    bag_np = bag_dataframe_to_3d_numpy(bag_df, extract_dict)
    car1 = bag_np["car1"]
    car2 = bag_np["car2"]
    extras = bag_np["extras"]
    traj_lengths = bag_np["traj_lengths"]
    bag_idx = bag_np["bag_idx"]
    bag_names = bag_np["bag_names"]
    # doubling up the data/symmetry
    car1_augmented = np.concatenate((car1, car2), axis=0)
    car2_augmented = np.concatenate((car2, car1), axis=0)
    extras_augmented = np.concatenate((extras, extras), axis=0)
    traj_lengths_augmented = np.concatenate((traj_lengths, traj_lengths), axis=0)
    bag_idx_augmented = np.concatenate((bag_idx, bag_idx), axis=0)

    # car1 = robot
    # car2 = human
    
    data_dict = {}
    data_dict["input_dict"] = {
        "car1": car1_augmented,
        "car2": car2_augmented,
        "extras": extras_augmented,
        "traj_lengths": traj_lengths_augmented,
        "bag_idx": bag_idx_augmented
    }
    data_dict["bag_names"] = bag_names
    data_dict["labels"] = data_dict["input_dict"]["car2"][:,:,pred_indices]
    data_dict["cars_mean"], data_dict["cars_std"] = extract_mean_and_std(data_dict["input_dict"]["car1"],
                                                                         data_dict["input_dict"]["traj_lengths"])
    data_dict["human_mean"], data_dict["human_std"] = extract_mean_and_std(data_dict["input_dict"]["car2"],
                                                                         data_dict["input_dict"]["traj_lengths"])
    data_dict["extras_mean"], data_dict["extras_std"] = extract_mean_and_std(data_dict["input_dict"]["extras"],
                                                                             data_dict["input_dict"]["traj_lengths"])
    data_dict["labels_mean"], data_dict["labels_std"] = extract_mean_and_std(data_dict["labels"],
                                                                             data_dict["input_dict"]["traj_lengths"])
    data_dict["pred_indices"] = pred_indices

    return data_dict

def split_bag_df(all_bags, train_eval_test_split = [.85, .15, 0.], seed = 1234):
    all_bag_names = list(all_bags.index.levels[0])
    np.random.seed(seed)
    np.random.shuffle(all_bag_names)

    train_cutoff, eval_cutoff, test_cutoff = map(int, np.cumsum(train_eval_test_split)*len(all_bag_names))

    train_bags = all_bags.loc[all_bag_names[:train_cutoff]]
    train_bags.set_index(pd.MultiIndex.from_tuples(train_bags.index.values), inplace=True)

    eval_bags = all_bags.loc[all_bag_names[train_cutoff:eval_cutoff]]
    eval_bags.set_index(pd.MultiIndex.from_tuples(eval_bags.index.values), inplace=True)

    if eval_cutoff == test_cutoff:
        test_bags = None
    else:
        test_bags = all_bags.loc[all_bag_names[eval_cutoff:]]
        test_bags.set_index(pd.MultiIndex.from_tuples(test_bags.index.values), inplace=True)

    return train_bags, eval_bags, test_bags

def load_split_data(filename="data/trajectories_full.pkl"):
    all_bags = pd.read_pickle(filename)
    train_bags, eval_bags, test_bags = split_bag_df(all_bags, train_eval_test_split = [.8, .2, 0.], seed=123) 
    
    return train_bags, eval_bags, test_bags, all_bags

### NEURAL NETWORK UTILS
def dropout_rnn_cell(cell_class, cell_kwargs, num_units, keep_prob, mode):
    if mode != tf.estimator.ModeKeys.TRAIN:
        cell_kwargs = cell_kwargs.copy()
        cell_kwargs.pop("dropout_keep_prob", None)
    cell = cell_class(num_units, **cell_kwargs)
    if mode == tf.estimator.ModeKeys.PREDICT and cell_class == tf.contrib.rnn.LSTMBlockCell:
        cell._names["scope"] = "layer_norm_basic_lstm_cell"    # HORRIBLE EXPORT HACK :/
    if mode == tf.estimator.ModeKeys.TRAIN and keep_prob < 1:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = keep_prob, output_keep_prob = keep_prob)
    return cell

def stacked_rnn_cell(cell_class, cell_kwargs, num_units_list, keep_prob, mode):
    num_layers = len(num_units_list)
    return tf.nn.rnn_cell.MultiRNNCell([dropout_rnn_cell(cell_class,
                                                         cell_kwargs,
                                                         num_units_list[layer],
                                                         keep_prob,
                                                         mode) for layer in range(num_layers)])

def project_to_RNN_initial_state(cell, input_tensor, scope="initial_state_projection"):
    cell_state_size = cell.state_size
    return project_to_RNN_initial_state_helper(cell_state_size, input_tensor, scope)

def project_to_RNN_initial_state_helper(cell_state_size, input_tensor, scope="initial_state_projection"):
    with tf.variable_scope(scope):
        if isinstance(cell_state_size, tf.nn.rnn_cell.LSTMStateTuple):
            initial_c = tf.layers.dense(input_tensor, cell_state_size.c, activation=tf.nn.tanh, name="initial_c")
            initial_h = tf.layers.dense(input_tensor, cell_state_size.h, activation=tf.nn.tanh, name="initial_h")
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
        elif isinstance(cell_state_size, int):
            initial_state = tf.layers.dense(input_tensor, cell_state_size, activation=tf.nn.tanh, name="initial_state")
        elif isinstance(cell_state_size, tuple):
            initial_state = tuple(project_to_RNN_initial_state_helper(cs, input_tensor, "cell_" + str(i)) for i, cs in enumerate(cell_state_size))
        else:
            raise(Exception("Unknown rnn_cell.state_size!"))
        return initial_state

def unpack_RNN_state(state):
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        return tf.concat([unpack_RNN_state(state.c), unpack_RNN_state(state.h)], axis=1)
    elif isinstance(state, tuple):
        if len(state) > 0:
            return tf.concat([unpack_RNN_state(s) for s in state], axis=1)
        else:
            return unpack_RNN_state(state[0])
    return state

def extract_subtensor_per_batch_element(tensor, indices):
    batch_range = tf.range(tf.shape(tensor)[0])
    gather_indices = tf.stack([batch_range, indices], axis=1)
    return tf.gather_nd(tensor, gather_indices)

def cartesian_product_over_batch(x, y, name=None):
    x_tiled = tf.tile(tf.expand_dims(x, 0), [tf.shape(y)[0]] + [1]*len(x.shape))
    y_tiled = tf.tile(tf.expand_dims(y, 1), [1] + [tf.shape(x)[0]] + [1]*(len(x.shape)-1))
    return tf.reshape(tf.stack([x_tiled, y_tiled], -1),
                      [-1] + [d.value for d in x.shape[1:-1]] + [x.shape[-1].value + y.shape[-1].value],
                      name=name)

class GMM2D(object):

    def __init__(self, log_pis, mus, log_sigmas, corrs, clip_lo = -10, clip_hi = 10):
        # input shapes
        # pis: [..., GMM_c]
        # mus: [..., GMM_c*2]
        # sigmas: [..., GMM_c*2]
        # corrs: [..., GMM_c]
        GMM_c = log_pis.shape[-1].value

        # Sigma = [s1^2    p*s1*s2      L = [s1   0
        #          p*s1*s2 s2^2 ]            p*s2 sqrt(1-p^2)*s2]
        log_pis = log_pis - tf.reduce_logsumexp(log_pis, -1, keep_dims=True)
        mus = self.reshape_to_components(mus, GMM_c)                  # [..., GMM_c, 2]
        log_sigmas = self.reshape_to_components(tf.clip_by_value(log_sigmas, clip_lo, clip_hi), GMM_c)
        sigmas = tf.exp(log_sigmas)                                   # [..., GMM_c, 2]
        one_minus_rho2 = 1 - tf.square(corrs)                         # [..., GMM_c]
        # Ls = tf.stack([(sigmas*tf.stack([tf.ones_like(corrs), corrs], -1)),                           # val = [s1, p*s2]
        #                (sigmas*tf.stack([tf.zeros_like(corrs), tf.sqrt(one_minus_rho2)], -1))], # val = [0, sqrt(1-p^2)*s2]
        #               axis=-1)    # [..., GMM_c, 2, 2]
        self.L1 = sigmas*tf.stack([tf.ones_like(corrs), corrs], -1)                       # [..., GMM_c, 2] (column 1 of L)
        self.L2 = sigmas*tf.stack([tf.zeros_like(corrs), tf.sqrt(one_minus_rho2)], -1)    # [..., GMM_c, 2] (column 2 of L)

        self.batch_shape = log_pis.shape[:-1]
        self.GMM_c = GMM_c
        self.log_pis = log_pis        # [..., GMM_c]
        self.mus = mus                # [..., GMM_c, 2]
        self.log_sigmas = log_sigmas  # [..., GMM_c, 2]
        self.sigmas = sigmas          # [..., GMM_c, 2]
        self.corrs = corrs            # [..., GMM_c]
        self.one_minus_rho2 = one_minus_rho2  # [..., GMM_c]
        self.cat = distributions.Categorical(logits=log_pis)
        # self.MVN = distributions.MultivariateNormalTriL(mus, Ls)

    def sample(self):
        # MVN_samples = self.MVN.sample()    # [..., GMM_c, 2]
        MVN_samples = self.mus + (self.L1*tf.expand_dims(tf.random_normal(tf.shape(self.corrs)), -1) +   # [..., GMM_c, 2]
                                  self.L2*tf.expand_dims(tf.random_normal(tf.shape(self.corrs)), -1))    # (manual 2x2 matmul)
        cat_samples = self.cat.sample()    # [...]
        selector = tf.expand_dims(tf.one_hot(cat_samples, self.GMM_c), -1)
        return tf.reduce_sum(MVN_samples*selector, -2)

    def log_prob(self, x):
        # x: [..., 2]
        x = tf.expand_dims(x, -2)    # [..., 1, 2]
        dx = x - self.mus            # [..., GMM_c, 2]
        z = (tf.reduce_sum(tf.square(dx/self.sigmas), -1) -
             2*self.corrs*tf.reduce_prod(dx, -1)/tf.reduce_prod(self.sigmas, -1))    # [..., GMM_c]
        component_log_p = -(tf.log(self.one_minus_rho2) + 2*tf.reduce_sum(self.log_sigmas, -1) +
                            z/self.one_minus_rho2 +
                            2*np.log(2*np.pi))/2
        return tf.reduce_logsumexp(self.log_pis + component_log_p, -1)

    def reshape_to_components(self, tensor, GMM_c):
        return tf.reshape(tensor, [-1 if s.value is None else s.value for s in tensor.shape[:-1]] + [GMM_c, 2])

def GMM2Dslow(log_pis, mus, log_sigmas, corrs, clip_lo = -10, clip_hi = 10):
    # shapes
    # pis: [..., GMM_c]
    # mus: [..., GMM_c*state_dim]
    # sigmas: [..., GMM_c*state_dim]
    # corrs: [..., GMM_c]
    GMM_c = log_pis.shape[-1]

    mus_split = tf.split(mus, GMM_c, axis=-1)
    sigmas = tf.exp(tf.clip_by_value(log_sigmas, clip_lo, clip_hi))

    # Sigma = [s1^2    p*s1*s2      L = [s1   0
    #          p*s1*s2 s2^2 ]            p*s2 sqrt(1-p^2)*s2]
    sigmas_reshaped = tf.reshape(sigmas, [-1 if s.value is None else s.value for s in sigmas.shape[:-1]] + [GMM_c.value,2])
    Ls = tf.stack([(sigmas_reshaped*tf.stack([tf.ones_like(corrs), corrs], -1)),                      # [s1, p*s2]
                   (sigmas_reshaped*tf.stack([tf.zeros_like(corrs), tf.sqrt(1 - corrs**2)], -1))],    # [0, sqrt(1-p^2)*s2]
                  axis=-1)
    Ls_split = tf.unstack(Ls, axis=-3)

    cat = distributions.Categorical(logits=log_pis)
    dists = [distributions.MultivariateNormalTriL(mu, L) for mu, L in zip(mus_split, Ls_split)]
    return distributions.Mixture(cat, dists)

def GMMdiag(log_pis, mus, log_sigmas, clip_lo = -10, clip_hi = 10):
    # shapes
    # pis: [..., GMM_c]
    # mus: [..., GMM_c*state_dim]
    # sigmas: [..., GMM_c*state_dim]
    GMM_c = log_pis.shape[-1]
    ax = len(mus.shape) - 1

    mus_split = tf.split(mus, GMM_c, axis=ax)
    sigmas = tf.exp(tf.clip_by_value(log_sigmas, clip_lo, clip_hi))
    sigmas_split = tf.split(sigmas, GMM_c, axis=ax)

    cat = distributions.Categorical(logits=log_pis)
    dists = [distributions.MultivariateNormalDiag(mu, sigma) for mu, sigma in zip(mus_split, sigmas_split)]
    return distributions.Mixture(cat, dists)

def project_to_GMM_params(tensor, GMM_c, GMM_dim, MLP_dims=None):
    MLP_output = MLP(tensor, MLP_dims, tf.nn.relu, 1, None)
    log_pis = tf.layers.dense(MLP_output, GMM_c, name="log_pi_projection")
    mus = tf.layers.dense(MLP_output, GMM_c*GMM_dim, name="mu_projection")
    log_sigmas = tf.layers.dense(MLP_output, GMM_c*GMM_dim, name="log_sigma_projection")
    corrs = tf.nn.tanh(tf.layers.dense(MLP_output, GMM_c, name="corr_projection"))
    return log_pis, mus, log_sigmas, corrs

# TODO: BN!!!
def MLP(input_tensor, h_dims, activation, keep_prob, mode, batch_norm=True):
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout_rate = 1 - keep_prob
    else:
        dropout_rate = 0
    h = input_tensor
    if h_dims:
        for j, d in enumerate(h_dims):
            h = tf.layers.dropout(tf.layers.dense(h, d, activation=activation, name = "dense" + str(j)),
                                  rate = dropout_rate, name = "dropout" + str(j))
    return h

def exp_anneal(start, finish, rate):
    step = tf.train.get_global_step()
    return finish - (finish - start)*tf.pow(rate, tf.to_float(step))

def sigmoid_anneal(start, finish, center_step, steps_lo_to_hi):
    step = tf.train.get_global_step()
    return start + (finish - start)*tf.sigmoid(tf.to_float(step - center_step) * tf.to_float(1/steps_lo_to_hi))

def standardize(tensor, mean, std):
    if tensor.shape[-1] > 0:
        tile_ct = int(tensor.shape[-1].value / mean.shape[-1].value)
        return (tensor - tf.tile(mean, [tile_ct])) / tf.tile(std, [tile_ct])
    else:
        return tensor

def unstandardize(tensor, mean, std, include_bias=True):
    if tensor.shape[-1] > 0:
        tile_ct = int(tensor.shape[-1].value / mean.shape[-1].value)
        if include_bias:
            return tensor * tf.tile(std, [tile_ct]) + tf.tile(mean, [tile_ct])
        else:
            return tensor * tf.tile(std, [tile_ct])
    else:
        return tensor

def all_one_hot_combinations(N, K, dtype):
    return np.eye(K, dtype=dtype).take(np.reshape(np.indices([K]*N), [N,-1]).T, axis=0).reshape(-1,N*K)    # [K**N, N*K]

###

class CVAELatent(object):

    def __init__(self):
        pass

class MVGLatent(CVAELatent):

    def __init__(self, latent_dim, node):
        super(MVGLatent, self).__init__()
        self.node = node
        self.z_dim = latent_dim
        self.p_dist = None          # will be filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None          # will be filled in by MultimodalGenerativeCVAE.encoder

    def dist_from_h(self, h, mode):
        mu =  tf.layers.dense(h, self.z_dim, name="mu_projection")
        log_sigma = tf.layers.dense(h, 1, name="log_sigma_projection")
        sigma = tf.exp(tf.clip_by_value(log_sigma, -10, 10))
        return distributions.MultivariateNormalDiag(mu, sigma*tf.ones(self.z_dim))

    def sample_q(self, k, mode):
        return self.q_dist.sample(k)

    def sample_p(self, k, mode):
        return self.p_dist.sample(k)

    def kl_q_p(self):
        return tf.reduce_mean(distributions.kl_divergence(self.q_dist, self.p_dist))

    def q_log_prob(self, z):
        return self.q_dist.log_prob(z)    # [k, nbs]

    def p_log_prob(self, z):
        return self.p_dist.log_prob(z)    # [k, nbs]

    def get_p_dist_params(self):
        return tf.zeros([1])

    def summarize_for_tensorboard(self):
        q_mu, p_mu = self.q_dist.mean(), self.p_dist.mean()
        q_sigma, p_sigma = self.q_dist.stddev(), self.p_dist.stddev()
        tf.summary.histogram("mu_norm_diff", tf.norm(q_mu - p_mu, axis=1), family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("sigma_diff", q_sigma - p_sigma, family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("q_sigma", q_sigma, family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("p_sigma", p_sigma, family=self.node.name.replace(' ', '_'))

class DiscreteLatent(CVAELatent):

    def __init__(self, N, K, node):
        super(DiscreteLatent, self).__init__()
        self.node = node
        self.z_dim = N*K
        self.N = N
        self.K = K
        self.temp = None            # will be filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = None    # may be filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.p_dist = None          # will be filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None          # will be filled in by MultimodalGenerativeCVAE.encoder
        self.z_NK = None            # will be filled in by MultimodalGenerativeCVAE.encoder (calling self.sample, below)
        self.kl_min = None          # will be filled in by MultimodalGenerativeCVAE.setup_model

    def dist_from_h(self, h, mode):
        logits = tf.layers.dense(h, self.z_dim, name="projection")
        logits_separated = tf.reshape(logits, [-1, self.N, self.K])
        logits_separated_mean_zero = logits_separated - tf.reduce_mean(logits_separated, axis=-1, keep_dims=True)
        if self.z_logit_clip is not None and mode == tf.estimator.ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = tf.clip_by_value(logits_separated_mean_zero, -c, c)
        else:
            logits = logits_separated_mean_zero
        return distributions.OneHotCategorical(logits=logits)

    def sample_q(self, k, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            z_dist = distributions.RelaxedOneHotCategorical(self.temp, logits=self.q_dist.logits)
            z_NK = z_dist.sample(k)
        elif mode ==  tf.estimator.ModeKeys.EVAL:
            z_NK = tf.to_float(self.q_dist.sample(k))
        return tf.reshape(z_NK, [k, -1, self.N*self.K])

    def sample_p(self, k, mode):
        if mode == tf.estimator.ModeKeys.PREDICT and self.K**self.N < 100:
            bs = tf.shape(self.p_dist.probs)[0]
            z_NK = tf.cond(tf.equal(k, 0),
                           lambda: tf.tile(all_one_hot_combinations(self.N,self.K,np.float32),
                                           multiples=[1,bs]),             # [K**N, bs*N*K]
                           lambda: tf.to_float(self.p_dist.sample(k)))    # [sample_ct, bs, N, K]
            k = tf.cond(tf.equal(k, 0),    # for reshape below (at this point k = input_sample_ct)
                        lambda: self.K**self.N,
                        lambda: k)
        else:
            z_NK = tf.to_float(self.p_dist.sample(k))
        return tf.reshape(z_NK, [k, -1, self.N*self.K])

    def kl_q_p(self):
        kl_separated = distributions.kl_divergence(self.q_dist, self.p_dist)        # [bs/nbs, N]
        kl_minibatch = tf.reduce_mean(kl_separated, 0, keep_dims=True)              # [1, N]
        tf.summary.scalar("true_kl", tf.reduce_sum(kl_minibatch), family=self.node.name.replace(' ', '_'))
        if self.kl_min > 0:
            kl_lower_bounded = tf.maximum(kl_minibatch, self.kl_min)
            kl = tf.reduce_sum(kl_lower_bounded)    # [], i.e., scalar
        else:
            kl = tf.reduce_sum(kl_minibatch)        # [], i.e., scalar
        return kl

    def q_log_prob(self, z):
        k = z.shape[0].value
        z_NK = tf.reshape(z, [k, -1, self.N, self.K])
        return tf.reduce_sum(self.q_dist.log_prob(z_NK), axis=2)  # [k, nbs]

    def p_log_prob(self, z):
        # k = z.shape[0].value
        k = tf.shape(z)[0]    # the above fails for mode = PREDICT
        z_NK = tf.reshape(z, [k, -1, self.N, self.K])
        return tf.reduce_sum(self.p_dist.log_prob(z_NK), axis=2)  # [k, nbs]

    def get_p_dist_params(self):
        return self.p_dist.probs

    def summarize_for_tensorboard(self):
        tf.summary.histogram("p_z_x", self.p_dist.probs, family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("q_z_xy", self.q_dist.probs, family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("p_z_x_logits", self.p_dist.logits, family=self.node.name.replace(' ', '_'))
        tf.summary.histogram("q_z_xy_logits", self.q_dist.logits, family=self.node.name.replace(' ', '_'))
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    tf.summary.histogram("q_z_xy_logit{0}{1}".format(i,j), self.q_dist.logits[:,i,j], family=self.node.name.replace(' ', '_'))