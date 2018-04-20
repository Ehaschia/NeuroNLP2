import torch
import torch.nn as nn
import math
from neuronlp2.nlinalg import logsumexp
import numpy as np
from ..utils import sequence_mask, reverse_padded_sequence
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Embedding


class lveg(nn.Module):
    def __init__(self, num_labels, gaussian_dim, word_size):
        super(lveg, self).__init__()

        self.num_labels = num_labels
        self.gaussian_dim = gaussian_dim

        self.trans_weight = Parameter(num_labels, num_labels)
        self.trans_p_mu = Parameter(num_labels, num_labels, gaussian_dim)
        self.trans_p_var = Parameter(num_labels, num_labels, gaussian_dim)
        self.trans_c_mu = Parameter(num_labels, num_labels, gaussian_dim)
        self.trans_c_var = Parameter(num_labels, num_labels, gaussian_dim)
        self.s_weight_em = Embedding(word_size, 1)
        self.s_mu_em = Embedding(word_size, gaussian_dim)
        self.s_var_em = Embedding(word_size, gaussian_dim)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal(self.trans_weight)
        nn.init.xavier_normal(self.trans_p_mu)
        nn.init.xavier_normal(self.trans_p_var)
        nn.init.xavier_normal(self.trans_c_mu)
        nn.init.xavier_normal(self.trans_c_var)
        nn.init.xavier_normal(self.s_weight)

    def forward(self, input, mask):
        torch.random.manual_seed(48)

        batch, length = input.size()

        s_mu = self.s_mu_em(input)
        s_weight = self.s_weight_em(input)
        s_var = self.s_var_em(input)

        t_weight = self.trans_weight.view(1, 1, self.num_labels, self.num_labels).expand(batch, length, self.num_labels, self.num_labels)
        t_p_mu = self.trans_p_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
        t_p_var = self.trans_p_var.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
        t_c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
        t_c_var = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            return scale, mu, var

        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu.unsqueeze(2), s_var.unsqueeze(2), t_c_mu, t_c_var)

        cs_scale = cs_scale + s_weight.unsqueeze(2)

        csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1, :, :, :].unsqueeze(4), cs_var[:, :-1, :, :, :].unsqueeze(4),
                                         t_p_mu[:, 1:, :, :, :].unsqueeze(2), t_p_var[:, 1:, :, :, :].unsqueeze(2))

        csp_scale = csp_scale + cs_scale[:, :-1, :, :].unsqueeze(4) + t_weight[:, 1:, :, :].unsqueeze(2)

        output = torch.cat((csp_scale, cs_scale[:, -1, :, :].unsqueeze(1).unsqueeze(4).expand(batch, 1, self.num_labels,
                                                                                              self.num_labels,
                                                                                              self.num_labels)), dim=1)
        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return output

    def loss(self, input, mask, target):
        batch, length, num_labels, _, _ = input.size()
        is_cuda = False
        if mask is not None:
            mask_transpose = mask.transpose(0, 1)
        else:
            mask_transpose = None

        energy_transpose = input.transpose(0, 1)

        target_transpose = target.transpose(0, 1)

        partition = None

        if is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(num_labels - 1)
            tgt_energy = torch.zeros(batch).cuda()
            holder = torch.zeros(batch).long().cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(num_labels - 1)
            tgt_energy = torch.zeros(batch)
            holder = torch.zeros(batch).long()

        for t in range(length):
            # shape = [batch, num_label, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                # partition shape [batch, num_label, num_label]
                partition = curr_energy[:, -1, :, :]
            else:
                # shape = [batch, num_label, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(3), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
                if t == length - 1:
                    partition = partition[:, :, 2]
            if t != length - 1:
                tgt_energy += curr_energy[
                    batch_index, prev_label, target_transpose[t], target_transpose[t + 1]]
                prev_label = target_transpose[t]
            else:
                tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t], holder]
                prev_label = target_transpose[t]
        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, energy, mask, score, leading_symbolic=0):
        is_cuda = False
        energy_transpose = energy.transpose(0, 1)
        mask_transpose = mask.transpose(0, 1)
        length, batch_size, num_label, _, _ = energy_transpose.size()

        # Forward word and Backward
        # Todo lexicon rule expected count

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask_transpose, batch_first=False)

        forward = torch.zeros([length-1, batch_size, num_label, num_label])
        backward = torch.zeros([length-1, batch_size, num_label, num_label])
        holder = torch.zeros([1, batch_size, num_label, num_label])

        for i in range(0, length - 1):
            if i == 0:
                forward[i] = energy_transpose[i, :, -1, :, :]
                backward[i] = reverse_energy_transpose[i, :, -1, :, :]
            else:
                forward[i] = logsumexp(forward[i - 1].unsqueeze(3) + energy_transpose[i], dim=1) \
                             * mask_transpose[i].unsqueeze(1).unsqueeze(2)
                backward[i] = logsumexp(backward[i - 1].unsqueeze(3) + reverse_energy_transpose[i], dim=1) \
                              * mask_transpose[i].unsqueeze(1).unsqueeze(2)
        backward = reverse_padded_sequence(backward, mask_transpose, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        cnt = forward + backward - score
        cnt_transpose = cnt[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = cnt_transpose.size()

        if is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label, 1])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        # pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        # viterbi docoding?
        pi[0] = cnt[0, :, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length - 1):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(cnt_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic
