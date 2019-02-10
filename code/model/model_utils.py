import torch
import torch.nn.utils.rnn as rnn
from enum import Enum
import functools


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def to_one_hot(labels, n_labels, device):
    return torch.eye(n_labels, device=device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, break_indices):
    pad_list = list()
    sorted_break_idxs, unsort_idxs = torch.sort(break_indices, descending=True)
    for i, seq_len in enumerate(sorted_break_idxs):
        pad_list.append(original_seqs[i, :seq_len])

    packed_seqs = rnn.pack_sequence(pad_list)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output, 
                                        batch_first=True,
                                        total_length=torch.max(break_indices))
    
    # Returning it in its original order.
    return output[unsort_idxs], (h_n[:, unsort_idxs], c_n[:, unsort_idxs])


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)
    
    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification: 
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
