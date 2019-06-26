import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np
from model.model_utils import to_one_hot, ModeKeys


class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()
        
        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim)/2)

        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)


    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))


    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])], dim=1)
        # score_vec is of shape (batch, num_enc_states)
        
        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)
        
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs


class TemporallyBatchedAdditiveAttention(AdditiveAttention):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(TemporallyBatchedAdditiveAttention, self).__init__(encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim)


    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        # return value should be of shape (batch, num_enc_states, max_time, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + torch.unsqueeze(self.w2(decoder_state), dim=1)))


    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, max_time, enc_dim)
        # decoder_state is of shape (batch, max_time, dec_dim)
        score_vec = self.score(encoder_states, decoder_state)
        # score_vec is of shape (batch, num_enc_states, max_time, 1)
        
        attention_probs = F.softmax(score_vec, dim=1)
        # attention_probs is of shape (batch, num_enc_states, max_time, 1)
        
        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, max_time, enc_dim)
        
        return final_context_vec, torch.squeeze(torch.transpose(attention_probs, 1, 2), dim=3)


class GMM2D(object):
    def __init__(self, log_pis, mus, log_sigmas, corrs, hyperparams, device,
                 clip_lo=-10, clip_hi=10):
        self.device = device
        self.hyperparams = hyperparams

        # input shapes
        # pis: [..., GMM_c]
        # mus: [..., GMM_c*2]
        # sigmas: [..., GMM_c*2]
        # corrs: [..., GMM_c]
        GMM_c = log_pis.shape[-1]

        # Sigma = [s1^2    p*s1*s2      L = [s1   0
        #          p*s1*s2 s2^2 ]            p*s2 sqrt(1-p^2)*s2]
        log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
        mus = self.reshape_to_components(mus, GMM_c)         # [..., GMM_c, 2]
        log_sigmas = self.reshape_to_components(torch.clamp(log_sigmas, min=clip_lo, max=clip_hi), GMM_c)
        sigmas = torch.exp(log_sigmas)                       # [..., GMM_c, 2]
        one_minus_rho2 = 1 - corrs**2                        # [..., GMM_c]

        self.L1 = sigmas*torch.stack([torch.ones_like(corrs, device=self.device), corrs], dim=-1)
        self.L2 = sigmas*torch.stack([torch.zeros_like(corrs, device=self.device), torch.sqrt(one_minus_rho2)], dim=-1)

        self.batch_shape = log_pis.shape[:-1]
        self.GMM_c = GMM_c
        self.log_pis = log_pis        # [..., GMM_c]
        self.mus = mus                # [..., GMM_c, 2]
        self.log_sigmas = log_sigmas  # [..., GMM_c, 2]
        self.sigmas = sigmas          # [..., GMM_c, 2]
        self.corrs = corrs            # [..., GMM_c]
        self.one_minus_rho2 = one_minus_rho2  # [..., GMM_c]
        self.cat = td.Categorical(logits=log_pis)


    def sample(self):
        MVN_samples = (self.mus 
                       + self.L1*torch.unsqueeze(torch.randn_like(self.corrs, device=self.device), dim=-1)     # [..., GMM_c, 2]
                       + self.L2*torch.unsqueeze(torch.randn_like(self.corrs, device=self.device), dim=-1))    # (manual 2x2 matmul)
        cat_samples = self.cat.sample()    # [...]
        selector = torch.unsqueeze(to_one_hot(cat_samples, self.GMM_c, self.device), dim=-1)
        return torch.sum(MVN_samples*selector, dim=-2)


    def log_prob(self, x):
        # x: [..., 2]
        x = torch.unsqueeze(x, dim=-2)    # [..., 1, 2]
        dx = x - self.mus            # [..., GMM_c, 2]
        z = (torch.sum((dx/self.sigmas)**2, dim=-1) -
             2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1))    # [..., GMM_c]
        component_log_p = -(torch.log(self.one_minus_rho2) + 2*torch.sum(self.log_sigmas, dim=-1) +
                            z/self.one_minus_rho2 +
                            2*np.log(2*np.pi))/2
        return torch.logsumexp(self.log_pis + component_log_p, dim=-1)


    def reshape_to_components(self, tensor, GMM_c):
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [GMM_c, self.hyperparams['pred_dim']])


def all_one_hot_combinations(N, K):
    return np.eye(K).take(np.reshape(np.indices([K]*N), [N,-1]).T, axis=0).reshape(-1, N*K)    # [K**N, N*K]


class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams['N'] * hyperparams['K']
        self.N = hyperparams['N']
        self.K = hyperparams['K']
        self.kl_min = hyperparams['kl_min']
        self.device = device
        self.temp = None            # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = None    # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.p_dist = None          # filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None          # filled in by MultimodalGenerativeCVAE.encoder


    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero
        
        if logits.size()[0] == 1:
            logits = torch.squeeze(logits, dim=0)
        
        return td.OneHotCategorical(logits=logits)


    def sample_q(self, k, mode):
        if mode == ModeKeys.TRAIN:
            z_dist = td.RelaxedOneHotCategorical(self.temp, logits=self.q_dist.logits)
            z_NK = z_dist.rsample((k, ))
        elif mode == ModeKeys.EVAL:
            z_NK = self.q_dist.sample((k, ))
        return torch.reshape(z_NK, (k, -1, self.z_dim))


    def sample_p(self, k, mode, most_likely=False):
        if mode == ModeKeys.PREDICT and self.K**self.N < 100 and k == 0:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(all_one_hot_combinations(self.N, self.K)).to(self.device).repeat(1, bs)
            k = self.K**self.N

        elif most_likely:
            # Sampling the most likely z from p(z|x).
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=1)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(k, -1, -1)

        else:
            z_NK = self.p_dist.sample((k, ))

        return torch.reshape(z_NK, (k, -1, self.N*self.K))


    def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        if len(kl_separated.size()) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)
            
        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)
        
        if log_writer is not None:
            log_writer.add_scalar(prefix + '/true_kl', torch.sum(kl_minibatch), curr_iter)

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)
        
        return kl


    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)


    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)


    def get_p_dist_probs(self):
        return self.p_dist.probs


    def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
        log_writer.add_histogram(prefix + "/latent/p_z_x", self.p_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy", self.q_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/p_z_x_logits", self.p_dist.logits, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy_logits", self.q_dist.logits, curr_iter)
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    log_writer.add_histogram(prefix + "/latent/q_z_xy_logit{0}{1}".format(i,j), self.q_dist.logits[:,i,j], curr_iter)
