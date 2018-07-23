import collections
from itertools import repeat
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
import numpy as np


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def prepare_rnn_seq(rnn_input, lengths, hx=None, masks=None, batch_first=False):
    '''

    Args:
        rnn_input: [seq_len, batch, input_size]: tensor containing the features of the input sequence.
        lengths: [batch]: tensor containing the lengthes of the input sequence
        hx: [num_layers * num_directions, batch, hidden_size]: tensor containing the initial hidden state for each element in the batch.
        masks: [seq_len, batch]: tensor containing the mask for each element in the batch.
        batch_first: If True, then the input and output tensors are provided as [batch, seq_len, feature].

    Returns:

    '''
    def check_decreasing(lengths):
        lens, order = torch.sort(lengths, dim=0, descending=True)
        if torch.ne(lens, lengths).sum() == 0:
            return None
        else:
            _, rev_order = torch.sort(order)
            return lens, Variable(order), Variable(rev_order)

    check_res = check_decreasing(lengths)

    if check_res is None:
        lens = lengths
        rev_order = None
    else:
        lens, order, rev_order = check_res
        batch_dim = 0 if batch_first else 1
        rnn_input = rnn_input.index_select(batch_dim, order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, order)
                cx = cx.index_select(1, order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, order)

    lens = lens.tolist()
    seq = rnn_utils.pack_padded_sequence(rnn_input, lens, batch_first=batch_first)
    if masks is not None:
        if batch_first:
            masks = masks[:, :lens[0]]
        else:
            masks = masks[:lens[0]]
    return seq, hx, rev_order, masks


def recover_rnn_seq(seq, rev_order, hx=None, batch_first=False):
    output, _ = rnn_utils.pad_packed_sequence(seq, batch_first=batch_first)
    if rev_order is not None:
        batch_dim = 0 if batch_first else 1
        output = output.index_select(batch_dim, rev_order)
        if hx is not None:
            # hack lstm
            if isinstance(hx, tuple):
                hx, cx = hx
                hx = hx.index_select(1, rev_order)
                cx = cx.index_select(1, rev_order)
                hx = (hx, cx)
            else:
                hx = hx.index_select(1, rev_order)
    return output, hx


def check_numerics(input, position=None):
    if input is None:
        return
    if input.is_cuda:
        check_res = np.isfinite(input.data.cpu().numpy())
    else:
        check_res = np.isfinite(input.data.numpy())
    check_res = np.subtract(check_res, 1.0)
    if np.sum(check_res) != 0.0:
        print("Numerics Error!")
        idx = np.where(check_res != 0.0)
        if position is not None:
            print(position)
        print("Idx:\t " + str(idx))
        exit(1)
    # check_big = 1.0*np.greater(np.abs(input.data.cpu().numpy()), 1e6)
    # if np.sum(check_big) != 0.0:
    #     print("Too big error!")
    #     exit(1)


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    # seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda(sequence_length.get_device())
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    # change for debug test_lveg
    if lengths.is_cuda:
        lengths = torch.sum(lengths, dim=1).cpu().numpy().astype(int)
    else:
        lengths = torch.sum(lengths, dim=1).numpy().astype(int)
    # if lengths.is_cuda:
    #     lengths = torch.sum(lengths, dim=0).cpu().numpy().astype(int)
    # else:
    #     lengths = torch.sum(lengths, dim=0).numpy().astype(int)
    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]

    input_size = inputs.size()
    inputs = inputs.contiguous()
    inputs = inputs.view(input_size[0], input_size[1], -1)

    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    # reversed_indices = Variable(reversed_indices)
    if inputs.is_cuda:
        device = inputs.get_device()
        reversed_indices = reversed_indices.cuda(device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    reversed_inputs = reversed_inputs.view(input_size)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs
