import torch
import torch.nn as nn
import math
from neuronlp2.nlinalg import logsumexp
import numpy as np
from neuronlp2.nn.utils import sequence_mask, reverse_padded_sequence
from torch.nn.parameter import Parameter
from torch.nn import Embedding
from neuronlp2.io import get_logger, conllx_data
import time
import sys
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from neuronlp2.nn.utils import check_numerics
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class lveg2(nn.Module):
    def __init__(self, word_size, num_labels, gaussian_dim=1, component=1):
        super(lveg2, self).__init__()

        self.num_labels = num_labels + 1
        self.gaussian_dim = gaussian_dim
        self.component = component
        self.min_clip = -5.0
        self.max_clip = 5.0
        self.trans_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component))
        self.trans_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.s_weight_em = Embedding(word_size, self.num_labels * self.component)
        self.s_mu_em = Embedding(word_size, self.num_labels * gaussian_dim * self.component)
        self.s_var_em = Embedding(word_size, self.num_labels * gaussian_dim * self.component)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_weight)
        nn.init.xavier_normal_(self.trans_p_mu)
        nn.init.xavier_normal_(self.trans_p_var)
        nn.init.xavier_normal_(self.trans_c_mu)
        nn.init.xavier_normal_(self.trans_c_var)

    def forward(self, input, mask):

        batch, length = input.size()

        s_mu = self.s_mu_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)
        s_weight = self.s_weight_em(input).view(batch, length, self.num_labels, self.component)
        # s_weight = Variable(torch.zeros(batch, length, self.num_labels)).cuda()
        s_var = self.s_var_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)

        t_weight = self.trans_weight.view(1, 1, self.num_labels, self.num_labels, self.component)
        # t_weight = t_weight.expand(batch, length, self.num_labels, self.num_labels, self.component)
        # t_weight = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels)).cuda()
        t_p_mu = self.trans_p_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        # t_p_mu = t_p_mu.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_var = self.trans_p_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        # t_p_var = t_p_var.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        # t_c_mu = t_c_mu.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_var = self.trans_c_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        # t_c_var = t_c_var.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)

        # s_mu = F.tanh(s_mu)
        # s_var = F.tanh(s_var)
        # t_p_mu = F.tanh(t_p_mu)
        # t_p_var = F.tanh(t_p_var)
        # t_c_mu = F.tanh(t_c_mu)
        # t_c_var = F.tanh(t_c_var)

        # s_mu = torch.clamp(s_mu, min=self.min_clip, max=self.max_clip)
        # s_var = torch.clamp(s_var, min=self.min_clip, max=self.max_clip)
        # t_p_mu = torch.clamp(t_p_mu, min=self.min_clip, max=self.max_clip)
        # t_p_var = torch.clamp(t_p_var, min=self.min_clip, max=self.max_clip)
        # t_c_mu = torch.clamp(t_c_mu, min=self.min_clip, max=self.max_clip)
        # t_c_var = torch.clamp(t_c_var, min=self.min_clip, max=self.max_clip)
        check_numerics(s_weight)
        check_numerics(s_mu)
        check_numerics(s_var)
        check_numerics(t_weight)
        check_numerics(t_p_mu)
        check_numerics(t_p_var)
        check_numerics(t_c_mu)
        check_numerics(t_c_var)

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input  shape1 [batch, length, num_labels, component 0, ... , component k-1, 1, gaussian_dim]
            # input  shape2 [batch, length, num_labels, 1, ..., 1, , component k, gaussian_dim]
            # output shape  [batch, length, num_labels, component 0, ... , component k-1, component k, gaussian_dim]
            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            check_numerics(n1_var_square)
            check_numerics(n2_var_square)
            check_numerics(var_square_add)
            check_numerics(var_log_square_add)
            check_numerics(scale)
            check_numerics(mu)
            check_numerics(var)
            return scale, mu, var

        # shape [batch, length, num_labels, num_labels, component, component, gaussian_dim]
        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu.unsqueeze(2).unsqueeze(4), s_var.unsqueeze(2).unsqueeze(4),
                                                 t_c_mu.unsqueeze(5), t_c_var.unsqueeze(5))

        cs_scale = cs_scale + s_weight.unsqueeze(2).unsqueeze(4)
        check_numerics(cs_scale)
        # shape [batch, length-1, num_labels, num_labels, num_labels, component, component, component, gaussian_dim]
        cs_mu.retain_grad()
        cs_var.retain_grad()
        t_p_mu.retain_grad()
        t_p_var.retain_grad()

        csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                         cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                         t_p_mu.unsqueeze(2).unsqueeze(5).unsqueeze(6),
                                         t_p_var.unsqueeze(2).unsqueeze(5).unsqueeze(6))

        csp_scale = csp_scale + cs_scale[:, :-1].unsqueeze(4).unsqueeze(7) + t_weight.unsqueeze(2).unsqueeze(
            5).unsqueeze(6)
        # output shape [batch, length, num_labels, num_labels, num_labels, component, component, component]
        check_numerics(csp_scale)

        return csp_scale, cs_mu, cs_var, t_p_mu, t_p_var

        output = torch.cat(
            (csp_scale, cs_scale[:, -1].unsqueeze(1).unsqueeze(4).unsqueeze(7).expand(batch, 1, self.num_labels,
                                                                                      self.num_labels, self.num_labels,
                                                                                      self.component, self.component,
                                                                                      self.component)), dim=1)
        if mask is not None:
            # maybe here is error
            output = output * mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6).unsqueeze(7)
        return output

    def loss(self, sents, target, mask):
        # fixme not calculate the toy sample by hand, because is too hard
        batch, length = sents.size()
        energy = self.forward(sents, mask)
        # for debug
        return torch.sum(energy[0]), energy

        if mask is not None:
            mask_transpose = mask.transpose(0, 1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            mask_transpose = None

        energy_transpose = energy.transpose(0, 1)

        target_transpose = target.transpose(0, 1)

        partition = None

        if energy.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, self.component, requires_grad=True).cuda()
            holder = torch.zeros(batch).long().cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, self.component, requires_grad=True)
            holder = torch.zeros(batch).long()

        for t in range(length):
            # shape = [batch, num_label, num_label, num_label]
            curr_energy = energy_transpose[t]
            check_numerics(partition)
            check_numerics(tgt_energy)
            if t == 0:
                # partition shape [batch, num_label, num_label, component, component, component]
                partition = curr_energy[:, -1]
                # shape [batch, num_label, num_label, component]
                partition = logsumexp(logsumexp(partition, dim=4), dim=3)
            else:
                # shape = [batch, num_label, num_label, num_label ,component, component, component]
                partition_new = logsumexp(logsumexp(logsumexp(curr_energy + partition.unsqueeze(3).unsqueeze(5).
                                                              unsqueeze(6), dim=5), dim=4), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
                if t == length - 1:
                    partition = partition[:, :, 2]
            if t != length - 1:
                tmp_energy = curr_energy[
                    batch_index, prev_label, target_transpose[t], target_transpose[t + 1]]
            else:
                tmp_energy = curr_energy[
                    batch_index, prev_label, target_transpose[t], holder]
            # tgt_energy = logsumexp(logsumexp(tmp_energy + tgt_energy.unsqueeze(2).unsqueeze(3), dim=2), dim=1)
            tgt_energy_new = logsumexp(logsumexp(tmp_energy + tgt_energy.unsqueeze(2).unsqueeze(3), dim=2), dim=1)
            if mask_transpose is None or t is 0:
                tgt_energy = tgt_energy_new
            else:
                mask_t = mask_transpose[t]
                tgt_energy = tgt_energy + (tgt_energy_new - tgt_energy) * mask_t.squeeze(3).squeeze(2)
            prev_label = target_transpose[t]
        loss = logsumexp(logsumexp(partition, dim=2), dim=1) - logsumexp(tgt_energy, dim=1)
        return loss.mean()

    def decode(self, sents, target, mask, lengths, leading_symbolic=0):
        is_cuda = True
        energy = self.forward(sents, mask).data
        energy_transpose = energy.transpose(0, 1)
        # fixme convert mask to Tensor
        mask = mask.data
        mask_transpose = mask.transpose(0, 1)
        length, batch_size, num_label, _, _, _, _, _ = energy_transpose.size()

        # Forward word and Backward

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask, batch_first=False)

        forward = torch.zeros([length - 1, batch_size, num_label, num_label, self.component]).cuda()
        backward = torch.zeros([length - 1, batch_size, num_label, num_label, self.component]).cuda()
        holder = torch.zeros([1, batch_size, num_label, num_label, self.component]).cuda()
        mask_transpose = mask_transpose.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # fixme version 2  remove leading_symbolic before expect_count
        for i in range(0, length - 1):
            if i == 0:
                forward[i] = logsumexp(logsumexp(energy_transpose[i, :, -1], dim=4), dim=3)
                backward[i] = logsumexp(logsumexp(reverse_energy_transpose[i, :, :, :, 2], dim=5), dim=4)
            else:
                forward[i] = logsumexp(logsumexp(
                    logsumexp(forward[i - 1].unsqueeze(3).unsqueeze(5).unsqueeze(6) + energy_transpose[i], dim=5),
                    dim=4), dim=1)
                forward[i] = forward[i - 1] + (forward[i] - forward[i - 1]) * mask_transpose[i]
                # backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3) \
                #               * mask_transpose[i].unsqueeze(1).unsqueeze(2)
                backward[i] = logsumexp(logsumexp(
                    logsumexp(backward[i - 1].unsqueeze(1).unsqueeze(4).unsqueeze(5) + reverse_energy_transpose[i],
                              dim=6), dim=5), dim=3)
                backward[i] = backward[i - 1] + (backward[i] - backward[i - 1]) * mask_transpose[i]

        # detect score calculate by forward and backward, should be equal
        # it is right to be here?
        forward_score = logsumexp(logsumexp(forward[-1, :, :, 2], dim=-1), dim=1)
        backword_score = logsumexp(logsumexp(backward[-1, :, -1, :], dim=-1), dim=1)
        err = forward_score - backword_score

        backward = reverse_padded_sequence(backward.contiguous(), mask, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        cnt = logsumexp(forward + backward, dim=-1)
        cnt_transpose = cnt[:, :, :-1, :-1]

        length, batch_size, num_label, _ = cnt_transpose.size()

        if is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        # pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        # viterbi docoding?
        pi[0] = cnt[0, :, -1, :-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].unsqueeze(2)
            pi[t], pointer[t] = torch.max(cnt_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        preds = back_pointer.transpose(0, 1)
        if target is None:
            return preds, None
        # if lengths is not None:
        #     max_len = lengths.max()
        #     target = target[:, :max_len]
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()


class lveg(nn.Module):
    def __init__(self, word_size, num_labels, gaussian_dim=1, component=1):
        super(lveg, self).__init__()

        self.num_labels = num_labels
        self.gaussian_dim = gaussian_dim
        self.component = component
        self.min_clip = -5.0
        self.max_clip = 5.0
        self.trans_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.trans_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.component, gaussian_dim))
        self.s_mu_em = Embedding(word_size, self.num_labels * gaussian_dim * self.component)
        self.s_var_em = Embedding(word_size, self.num_labels * gaussian_dim * self.component)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_p_mu)
        nn.init.xavier_normal_(self.trans_p_var)
        nn.init.xavier_normal_(self.trans_c_mu)
        nn.init.xavier_normal_(self.trans_c_var)

    def forward(self, input):
        batch, length = input.size()

        s_mu = self.s_mu_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)
        s_var = self.s_var_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)

        t_p_mu = self.trans_p_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_mu = t_p_mu.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_var = self.trans_p_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_var = t_p_var.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_mu = t_c_mu.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_var = self.trans_c_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_c_var = t_c_var.expand(batch, length, self.num_labels, self.num_labels, self.component, self.gaussian_dim)

        check_numerics(s_mu)
        check_numerics(s_var)
        check_numerics(t_p_mu)
        check_numerics(t_p_var)
        check_numerics(t_c_mu)
        check_numerics(t_c_var)

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input  shape1 [batch, length, num_labels, component 0, ... , component k-1, 1, gaussian_dim]
            # input  shape2 [batch, length, num_labels, 1, ..., 1, , component k, gaussian_dim]
            # output shape  [batch, length, num_labels, component 0, ... , component k-1, component k, gaussian_dim]
            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            check_numerics(n1_var_square)
            check_numerics(n2_var_square)
            check_numerics(var_square_add)
            check_numerics(var_log_square_add)
            check_numerics(scale)
            check_numerics(mu)
            check_numerics(var)
            return scale, mu, var

        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu.unsqueeze(2).unsqueeze(4), s_var.unsqueeze(2).unsqueeze(4),
                                                 t_c_mu.unsqueeze(5), t_c_var.unsqueeze(5))

        check_numerics(cs_scale)
        cs_mu.retain_grad()
        cs_var.retain_grad()
        t_p_mu.retain_grad()
        t_p_var.retain_grad()

        csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                         cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                         t_p_mu[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6),
                                         t_p_var[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6))

        check_numerics(csp_scale)

        return csp_scale, cs_mu, cs_var, t_p_mu, t_p_var

    def loss(self, sents):
        energy = self.forward(sents)
        # for debug
        return torch.sum(energy[0]), energy


class DebugModule(nn.Module):
    def __init__(self, num_label, component, gaussian_dim):
        super(DebugModule, self).__init__()
        self.num_labels = num_label
        self.component = component
        self.gaussian_dim = gaussian_dim

        self.s_mu_em = Embedding(num_label, num_label * component * gaussian_dim)
        self.s_var_em = Embedding(num_label, num_label * component * gaussian_dim)

        self.trans_c_mu = Parameter(
            torch.Tensor(self.num_labels, self.num_labels, self.component, self.gaussian_dim))
        self.trans_c_var = Parameter(
            torch.Tensor(self.num_labels, self.num_labels, self.component, self.gaussian_dim))
        self.trans_p_mu = Parameter(torch.Tensor(num_label, num_label, component, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(num_label, num_label, component, gaussian_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.s_mu_em.weight)
        nn.init.uniform_(self.s_var_em.weight)
        nn.init.uniform_(self.trans_c_mu)
        nn.init.uniform_(self.trans_c_var)
        nn.init.uniform_(self.trans_p_mu)
        nn.init.uniform_(self.trans_p_var)

    def forward(self, input):
        batch, length = input.size()
        s_mu = self.s_mu_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)
        s_var = self.s_var_em(input).view(batch, length, self.num_labels, self.component, self.gaussian_dim)

        t_p_mu = self.trans_p_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_var = self.trans_p_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        t_p_mu = t_p_mu.expand(batch, length, self.num_labels, self.num_labels, self.component,
                               self.gaussian_dim)
        t_p_var = t_p_var.expand(batch, length, self.num_labels, self.num_labels, self.component,
                                 self.gaussian_dim)

        c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        c_var = self.trans_c_var.view(1, 1, self.num_labels, self.num_labels, self.component, self.gaussian_dim)
        c_mu = c_mu.expand(batch, length, self.num_labels, self.num_labels, self.component,
                           self.gaussian_dim)
        c_var = c_var.expand(batch, length, self.num_labels, self.num_labels, self.component,
                             self.gaussian_dim)
        t_p_mu.retain_grad()
        t_p_var.retain_grad()

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input  shape1 [batch, length, num_labels, component 0, ... , component k-1, 1, gaussian_dim]
            # input  shape2 [batch, length, num_labels, 1, ..., 1, , component k, gaussian_dim]
            # output shape  [batch, length, num_labels, component 0, ... , component k-1, component k, gaussian_dim]
            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            check_numerics(n1_var_square)
            check_numerics(n2_var_square)
            check_numerics(var_square_add)
            check_numerics(var_log_square_add)
            check_numerics(scale)
            check_numerics(mu)
            check_numerics(var)
            return scale, mu, var

        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu.unsqueeze(2).unsqueeze(4),
                                                 s_var.unsqueeze(2).unsqueeze(4),
                                                 c_mu.unsqueeze(5), c_var.unsqueeze(5))

        csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                         cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                         t_p_mu[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6),
                                         t_p_var[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6))
        return torch.sum(csp_scale), t_p_mu, t_p_var


class Minimalist_Mode(nn.Module):
    def __init__(self, batch, length, num_label, component, gaussian_dim):
        super(Minimalist_Mode, self).__init__()
        self.num_label = num_label
        self.component = component
        self.gaussian_dim = gaussian_dim

        self.cs_mu = Parameter(torch.Tensor(batch, length, num_label, num_label, component, component, gaussian_dim))
        self.cs_var = Parameter(torch.Tensor(batch, length, num_label, num_label, component, component, gaussian_dim))

        self.t_p_mu = Parameter(torch.Tensor(batch, length, num_label, num_label, component, gaussian_dim))
        self.t_p_var = Parameter(torch.Tensor(batch, length, num_label, num_label, component, gaussian_dim))

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.cs_mu)
        nn.init.uniform_(self.cs_var)
        nn.init.uniform_(self.t_p_mu)
        nn.init.uniform_(self.t_p_var)

    def forward(self):
        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input  shape1 [batch, length, num_labels, component 0, ... , component k-1, 1, gaussian_dim]
            # input  shape2 [batch, length, num_labels, 1, ..., 1, , component k, gaussian_dim]
            # output shape  [batch, length, num_labels, component 0, ... , component k-1, component k, gaussian_dim]
            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            check_numerics(n1_var_square)
            check_numerics(n2_var_square)
            check_numerics(var_square_add)
            check_numerics(var_log_square_add)
            check_numerics(scale)
            check_numerics(mu)
            check_numerics(var)
            return scale, mu, var

        csp_scale, _, _ = gaussian_multi(self.cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                         self.cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                         self.t_p_mu[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6),
                                         self.t_p_var[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6))

        return torch.sum(csp_scale)


class ExtendP(nn.Module):
    def __init__(self, batch, length, num_label, component, gaussian_dim):
        super(ExtendP, self).__init__()
        self.num_label = num_label
        self.component = component
        self.gaussian_dim = gaussian_dim
        self.batch = batch
        self.length = length

        self.cs_mu = Parameter(torch.Tensor(batch, length, num_label, num_label, component, component, gaussian_dim))
        self.cs_var = Parameter(torch.Tensor(batch, length, num_label, num_label, component, component, gaussian_dim))

        self.trans_p_mu = Parameter(torch.Tensor(num_label, num_label, component, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(num_label, num_label, component, gaussian_dim))

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.cs_mu)
        nn.init.uniform_(self.cs_var)
        nn.init.uniform_(self.trans_p_mu)
        nn.init.uniform_(self.trans_p_var)

    def forward(self):
        t_p_mu = self.trans_p_mu.unsqueeze(0).unsqueeze(0).expand(self.batch, self.length, self.num_label,
                                                                  self.num_label, self.component, self.gaussian_dim)

        t_p_var = self.trans_p_var.unsqueeze(0).unsqueeze(0).expand(self.batch, self.length, self.num_label,
                                                                    self.num_label, self.component, self.gaussian_dim)

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input  shape1 [batch, length, num_labels, component 0, ... , component k-1, 1, gaussian_dim]
            # input  shape2 [batch, length, num_labels, 1, ..., 1, , component k, gaussian_dim]
            # output shape  [batch, length, num_labels, component 0, ... , component k-1, component k, gaussian_dim]
            # n1_var_square = torch.exp(2.0 * n1_var)
            # n2_var_square = torch.exp(2.0 * n2_var)
            # var_square_add = n1_var_square + n2_var_square
            # var_log_square_add = torch.log(var_square_add)
            #
            # scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)
            #
            # mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add
            #
            # var = n1_var + n2_var - 0.5 * var_log_square_add
            # scale = torch.sum(scale, dim=-1)
            # check_numerics(n1_var_square)
            # check_numerics(n2_var_square)
            # check_numerics(var_square_add)
            # check_numerics(var_log_square_add)
            # check_numerics(scale)
            # check_numerics(mu)
            # check_numerics(var)

            scale = n1_mu + n1_var + n2_mu + n2_var
            return scale

        csp_scale = gaussian_multi(self.cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                   self.cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                   t_p_mu[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6),
                                   t_p_var[:, 1:].unsqueeze(2).unsqueeze(5).unsqueeze(6))

        return torch.sum(csp_scale)


def store(file, name, tensor):
    file.write(name)
    file.write(str(tensor.size()))
    file.write("\n")
    file.write(np.array2string(tensor.numpy(), precision=2, separator=',', suppress_small=True))
    file.write("\n\n")


def store_param(t_p_mu, t_p_var, network, v):
    with open("debug_param_" + v, 'w') as f:
        store(f, "t_p_mu:\n", t_p_mu.squeeze().data)
        store(f, "t_p_var:\n", t_p_var.squeeze().data)
        store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data)
        store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data)
        store(f, "trans_c_mu:\n", network.trans_c_mu.squeeze().data)
        store(f, "trans_c_var:\n", network.trans_c_var.squeeze().data)
        store(f, "s_mu_em:\n", network.s_mu_em.weight.squeeze().data)
        store(f, "s_var_em:\n", network.s_var_em.weight.squeeze().data)
    with open("debug_grad_" + v, 'w') as f:
        store(f, "t_p_mu:\n", t_p_mu.grad.squeeze().data)
        store(f, "t_p_var:\n", t_p_var.grad.squeeze().data)
        store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data)
        store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data)
        store(f, "trans_c_mu:\n", network.trans_c_mu.grad.squeeze().data)
        store(f, "trans_c_var:\n", network.trans_c_var.grad.squeeze().data)
        store(f, "s_mu_em:\n", network.s_mu_em.weight.grad.squeeze().data)
        store(f, "s_var_em:\n", network.s_var_em.weight.grad.squeeze().data)


def store_lveg_param(network, v, cs_mu, cs_var, t_p_mu, t_p_var, is_cuda):
    if is_cuda:
        with open("lveg_param_" + v, 'w') as f:
            store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data.cpu())
            store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data.cpu())
            store(f, "trans_c_mu:\n", network.trans_c_mu.squeeze().data.cpu())
            store(f, "trans_c_var:\n", network.trans_c_var.squeeze().data.cpu())
            store(f, "s_mu_em:\n", network.s_mu_em.weight.squeeze().data.cpu())
            store(f, "s_var_em:\n", network.s_var_em.weight.squeeze().data.cpu())
            store(f, "t_p_mu:\n", t_p_mu.squeeze().data.cpu())
            store(f, "t_p_var:\n", t_p_var.squeeze().data.cpu())
            store(f, "cs_mu:\n", cs_mu.squeeze().data.cpu())
            store(f, "cs_var:\n", cs_var.squeeze().data.cpu())
        with open("lveg_grad_" + v, 'w') as f:
            store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data.cpu())
            store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data.cpu())
            store(f, "trans_c_mu:\n", network.trans_c_mu.grad.squeeze().data.cpu())
            store(f, "trans_c_var:\n", network.trans_c_var.grad.squeeze().data.cpu())
            store(f, "s_mu_em:\n", network.s_mu_em.weight.grad.squeeze().data.cpu())
            store(f, "s_var_em:\n", network.s_var_em.weight.grad.squeeze().data.cpu())
            store(f, "t_p_mu:\n", t_p_mu.grad.squeeze().data.cpu())
            store(f, "t_p_var:\n", t_p_var.grad.squeeze().data.cpu())
            store(f, "cs_mu:\n", cs_mu.grad.squeeze().data.cpu())
            store(f, "cs_var:\n", cs_var.grad.squeeze().data.cpu())
    else:

        with open("lveg_param_" + v, 'w') as f:
            store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data)
            store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data)
            store(f, "trans_c_mu:\n", network.trans_c_mu.squeeze().data)
            store(f, "trans_c_var:\n", network.trans_c_var.squeeze().data)
            store(f, "s_mu_em:\n", network.s_mu_em.weight.squeeze().data)
            store(f, "s_var_em:\n", network.s_var_em.weight.squeeze().data)
            store(f, "t_p_mu:\n", t_p_mu.squeeze().data)
            store(f, "t_p_var:\n", t_p_var.squeeze().data)
            store(f, "cs_mu:\n", cs_mu.squeeze().data)
            store(f, "cs_var:\n", cs_var.squeeze().data)
        with open("lveg_grad_" + v, 'w') as f:
            store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data)
            store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data)
            store(f, "trans_c_mu:\n", network.trans_c_mu.grad.squeeze().data)
            store(f, "trans_c_var:\n", network.trans_c_var.grad.squeeze().data)
            store(f, "s_mu_em:\n", network.s_mu_em.weight.grad.squeeze().data)
            store(f, "s_var_em:\n", network.s_var_em.weight.grad.squeeze().data)
            store(f, "t_p_mu:\n", t_p_mu.grad.squeeze().data)
            store(f, "t_p_var:\n", t_p_var.grad.squeeze().data)
            store(f, "cs_mu:\n", cs_mu.grad.squeeze().data)
            store(f, "cs_var:\n", cs_var.grad.squeeze().data)


def store_minimalist_mode(network, v, is_cuda):
    if is_cuda:
        with open("minimalist_mode_param_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.squeeze().data.cpu())
            store(f, "cs_var:\n", network.cs_var.squeeze().data.cpu())
            store(f, "t_p_mu:\n", network.t_p_mu.squeeze().data.cpu())
            store(f, "t_p_var:\n", network.t_p_var.squeeze().data.cpu())
        with open("minimalist_mode_grad_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.grad.squeeze().data.cpu())
            store(f, "cs_var:\n", network.cs_var.grad.squeeze().data.cpu())
            store(f, "t_p_mu:\n", network.t_p_mu.grad.squeeze().data.cpu())
            store(f, "t_p_var:\n", network.t_p_var.grad.squeeze().data.cpu())
    else:
        with open("minimalist_mode_param_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.squeeze().data)
            store(f, "cs_var:\n", network.cs_var.squeeze().data)
            store(f, "t_p_mu:\n", network.t_p_mu.squeeze().data)
            store(f, "t_p_var:\n", network.t_p_var.squeeze().data)
        with open("minimalist_mode_grad_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.grad.squeeze().data)
            store(f, "cs_var:\n", network.cs_var.grad.squeeze().data)
            store(f, "t_p_mu:\n", network.t_p_mu.grad.squeeze().data)
            store(f, "t_p_var:\n", network.t_p_var.grad.squeeze().data)


def store_extendp(network, v, is_cuda):
    if is_cuda:
        with open("extendp_param_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.squeeze().data.cpu())
            store(f, "cs_var:\n", network.cs_var.squeeze().data.cpu())
            store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data.cpu())
            store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data.cpu())
        with open("extendp_grad_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.grad.squeeze().data.cpu())
            store(f, "cs_var:\n", network.cs_var.grad.squeeze().data.cpu())
            store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data.cpu())
            store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data.cpu())
    else:
        with open("extendp_param_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.squeeze().data)
            store(f, "cs_var:\n", network.cs_var.squeeze().data)
            store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data)
            store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data)
        with open("extendp_grad_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.grad.squeeze().data)
            store(f, "cs_var:\n", network.cs_var.grad.squeeze().data)
            store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data)
            store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data)


def main():
    batch, length, num_label, component, gaussian_dim = 4, 50, 50, 1, 1
    lr = 0.1
    momentum = 0.9
    gamma = 0.0
    is_cuda = True

    # model = DebugModule(num_label, component, gaussian_dim)
    # model = lveg(num_label, num_label, gaussian_dim=gaussian_dim, component=component)
    model = lveg2(num_label, num_label, gaussian_dim=gaussian_dim, component=component)
    # model = Minimalist_Mode(batch, length, num_label, component, gaussian_dim)
    # model = ExtendP(batch, length, num_label, component, gaussian_dim)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    sents = torch.randint(0, num_label, (batch, length)).long()

    if is_cuda:
        model.cuda()
        sents = sents.cuda()
    optim.zero_grad()
    # loss, t_p_mu, t_p_var = model(sents)
    loss, tmp_pru = model.loss(sents, None, None)
    # loss = model()
    loss.backward()
    # store_param(t_p_mu, t_p_var, model, 'simple_2')
    store_lveg_param(model, '13', tmp_pru[1], tmp_pru[2], tmp_pru[3], tmp_pru[4], is_cuda)
    # store_minimalist_mode(model, '19', is_cuda)
    # store_extendp(model, '9', is_cuda)
    optim.step()
    print(loss.item())


if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    main()
    # natural_data()
