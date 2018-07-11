import sys

sys.path.append(".")
sys.path.append("/public/sist/home/zhanglw/code/pos/0518/NeuroNLP2/")

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
import torch.nn.functional as F
from neuronlp2.nn.utils import check_numerics
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class ChainCRF(nn.Module):
    def __init__(self, word_size, num_labels, **kwargs):
        '''
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            **kwargs:
        '''
        super(ChainCRF, self).__init__()
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels

        self.state = Embedding(word_size, self.num_labels)
        self.trans_nn = None
        self.trans_matrix = Parameter(torch.Tensor(self.num_labels, self.num_labels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.trans_matrix)
        # if not self.bigram:
        #     nn.init.normal(self.trans_matrix)

    def forward(self, input, mask=None):
        '''
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]
        '''
        batch, length = input.size()

        # compute out_s by tensor dot [batch, length, input_size] * [input_size, num_label]
        # thus out_s should be [batch, length, num_label] --> [batch, length, num_label, 1]
        out_s = self.state(input).unsqueeze(2)

        # [batch, length, num_label, num_label]
        output = self.trans_matrix + out_s

        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)

        return output

    def loss(self, input, target, mask=None):
        '''
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]
        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        '''
        batch, length = input.size()
        energy = self.forward(input, mask=mask)
        # shape = [length, batch, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).transpose(0, 1)

        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, requires_grad=True).cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, requires_grad=True)

        for t in range(length):
            # shape = [batch, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                partition = curr_energy[:, -1, :]
            else:
                # shape = [batch, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(2), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t
            tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data]
            prev_label = target_transpose[t].data

        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, target, mask, lengths, leading_symbolic=0):
        """
        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            leading_symbolic: nt
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)
        Returns: Tensor
            decoding results in shape [batch, length]
        """

        energy = self.forward(input, mask=mask).data

        # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
        # For convenience, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
        energy_transpose = energy.transpose(0, 1)

        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        energy_transpose = energy_transpose[:, :, :-1, :-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        if input.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        pi[0] = energy[:, 0, -1, :-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].unsqueeze(2)
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev, dim=1)

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
            # return back_pointer.transpose(0, 1) + leading_symbolic


class lveg(nn.Module):
    def __init__(self, word_size, num_labels, bigram=False, spherical=False,
                 gaussian_dim=1, clip=1.0, k=1):
        """
            Only support e_comp = t_comp = 1 situation
        Args:
            input_size: int
                the dimension of the input.
            num_labels: int
                the number of labels of the crf layer
            bigram: bool
                if apply bi-gram parameter.
            spherical: bool
                if apply spherical gaussian
            gaussian_dim: int
                the dimension of gaussian
            clip: double
                clamp all elements into [-clip, clip]
            k: int
                pruning the inside score
        """
        super(lveg, self).__init__()
        self.word_size = word_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.spherical = spherical
        self.gaussian_dim = gaussian_dim
        self.min_clip = -clip
        self.max_clip = clip
        self.k = k
        # Gaussian for every emission rule
        # weight is log form
        # self.state_nn_weight = nn.Linear(self.input_size, self.num_labels * self.e_comp)
        self.s_weight_em = Embedding(word_size, self.num_labels)
        # every label  is a gaussian_dim dimension gaussian
        # self.state_nn_mu = nn.Linear(self.input_size,
        #                              self.num_labels * self.gaussian_dim * self.gaussian_dim * self.e_comp)
        # self.state_nn_var = nn.Linear(self.input_size,
        #                               self.num_labels * self.gaussian_dim * self.gaussian_dim * self.e_comp)
        self.s_mu_em = Embedding(word_size, self.num_labels * gaussian_dim)
        self.s_var_em = Embedding(word_size, self.num_labels * gaussian_dim * gaussian_dim)
        # weight and var is log form
        if self.bigram:
            self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels * self.num_labels)
            self.trans_nn_mu = nn.Linear(self.input_size,
                                         self.num_labels * self.num_labels * 2 * self.gaussian_dim)
            # (2*gaussian_dim * 2*gaussian_dim) matrix cholesky decomposition
            self.trans_nn_var = nn.Linear(self.input_size,
                                          self.num_labels * self.num_labels * (2*self.gaussian_dim+1)*self.gaussian_dim)

            self.register_parameter("trans_mat_weight", None)
            self.register_parameter("trans_mat_mu", None)
            self.register_parameter("trans_mat_var", None)

        else:
            self.trans_nn_weight = None
            self.trans_nn_p_mu = None
            self.trans_nn_c_mu = None
            self.trans_nn_p_var = None
            self.trans_nn_c_var = None

            self.trans_mat_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels))
            self.trans_mat_mu = Parameter(
                torch.Tensor(self.num_labels, self.num_labels, 2 * self.gaussian_dim))
            # (2*gaussian_dim * 2*gaussian_dim) matrix cholesky decomposition
            self.trans_mat_var = Parameter(
                torch.Tensor(self.num_labels, self.num_labels, (2*self.gaussian_dim+1)*self.gaussian_dim))
        self.reset_parameter()

    def reset_parameter(self):
        # nn.init.constant_(self.state_nn_weight.bias, 0)
        # nn.init.constant_(self.state_nn_var.bias, 0)
        # nn.init.constant_(self.state_nn_mu.bias, 0)

        if self.bigram:
            nn.init.xavier_normal_(self.trans_nn_weight.weight)
            nn.init.xavier_normal_(self.trans_nn_mu.weight)
            nn.init.xavier_normal_(self.trans_nn_var.weight)
            nn.init.constant_(self.trans_nn_weight.bias, 0)
            nn.init.constant_(self.trans_nn_p_mu.bias, 0)
            nn.init.constant_(self.trans_nn_p_var.bias, 0)
        else:
            nn.init.normal_(self.trans_mat_weight)
            nn.init.normal_(self.trans_mat_mu)
            nn.init.normal_(self.trans_mat_var)

    def inverse(self, input):
        # implement 1 dim and 2 dim
        epslion = 0.0
        size = input.size()
        if size[-1] != size[-2]:
            raise ValueError('To inverse matrix should be square.')
        if size[-1] == 1:
            return 1.0 / (input + epslion)
            # return -1.0 * input

        input = input.view(-1, size[-2], size[-1])
        if size[-1] == 2:
            # http://www.mathcentre.ac.uk/resources/uploaded/sigma-matrices7-2009-1.pdf
            scale = 1.0 / (input[:, 0, 0] * input[:, 1, 1] - input[:, 0, 1] * input[:, 1, 0] + epslion)
            # denoinator
            # deno = scale.view(scale.size() + (1, 1))
            inv_00 = (input[:, 1, 1] * scale).view(input.size()[0], 1)
            inv_01 = (- input[:, 1, 0] * scale).view(input.size()[0], 1)
            inv_10 = (- input[:, 0, 1] * scale).view(input.size()[0], 1)
            inv_11 = (input[:, 0, 0] * scale).view(input.size()[0], 1)

            inv = torch.cat((inv_00, inv_01, inv_10, inv_11), dim=1)
            return inv.view(size)
        else:
            raise NotImplementedError

    def log_inverse(self, input):
        return torch.log(self.inverse(torch.exp(input)))

    def determinate(self, input):
        epslion = 1e-5
        size = input.size()
        if size[-1] != size[-2]:
            raise NotImplementedError('Not square matrix determinate is not implement')
        if size[-1] == 1:
            return input.view(input.size()[:-2])
            # return -1.0 * input

        input = input.view(-1, size[-2], size[-1])
        if size[-1] == 2:
            # http://www.mathcentre.ac.uk/resources/uploaded/sigma-matrices7-2009-1.pdf
            det = input[:, 0, 0] * input[:, 1, 1] - input[:, 0, 1] * input[:, 1, 0]
            return det.view(size[:-2])
        else:
            raise NotImplementedError

    def pruning(self, score, mu, var, dim, merge_dims=None):
        if merge_dims is not None:
            score_dim = list(score.size())
            del score_dim[merge_dims[1]]
            score_dim[merge_dims[0]] = -1
            score_dim = tuple(score_dim)
            score = score.view(score_dim)
            mu = mu.view(score_dim)
            var = var.view(score_dim)
        pruned_score = torch.topk(score, self.k, dim)
        # this only support 1 dim situaltion
        pruned_var = torch.gather(var, dim, pruned_score[1])
        pruned_mu = torch.gather(mu, dim, pruned_score[1])
        # pruned_var = torch.gather(var, dim, pruned_score[1])
        return pruned_score[0].transpose(1, 2), pruned_mu.transpose(1, 2), pruned_var.transpose(1, 2)

    def forward(self, input, mask=None):

        batch, length = input.size()

        # compute out_weight, out_mu, out_var by tensor dot
        #
        # [batch, length, input_size] * [input_size, num_label*gaussian_dim]
        #
        # thus weight should be [batch, length, num_label*gaussian_dim] --> [batch, length, num_label, 1]
        #
        # the mu and var tensor should be [batch, length, num_label*gaussian_dim] -->
        # [batch, length, num_label, 1, gaussian_dim]
        #
        # if the s_var is spherical it should be [batch, length, 1, 1]

        # s_weight shape [batch, length, num_label]

        s_mu = self.s_mu_em(input).view(batch, length, 1, self.num_labels, self.gaussian_dim).transpose(0, 1)
        s_weight = self.s_weight_em(input).view(batch, length, 1, self.num_labels, 1).transpose(0, 1)
        s_var = self.s_var_em(input).view(batch, length, 1, self.num_labels, self.gaussian_dim,
                                          self.gaussian_dim).transpose(0, 1)
        # reset s_var
        s_var_size = s_var.size()
        s_var = s_var.contiguous()
        s_var = torch.bmm(s_var.view(-1, s_var_size[-2], s_var_size[-1]),
                          s_var.view(-1, s_var_size[-2], s_var_size[-1]).transpose(1, 2)).view(s_var_size)
        if self.bigram:
            # alert not debug
            t_weight = self.trans_nn_weight(input).view(batch, length, self.num_labels, self.num_labels, 1).transpose(0,
                                                                                                                      1)
            t_mu = self.trans_nn_p_mu(input).view(batch, length, self.num_labels, self.num_labels,
                                                  2 * self.gaussian_dim).transpose(0, 1)
            t_var = self.trans_nn_p_var(input).view(batch, length, self.num_labels, self.num_labels,
                                                    2 * self.gaussian_dim, 2 * self.gaussian_dim).transpose(0, 1)
        else:
            t_weight = self.trans_mat_weight.view(1, self.num_labels, self.num_labels, 1)
            t_mu = self.trans_mat_mu.view(1, self.num_labels, self.num_labels, 2 * self.gaussian_dim)
            t_var = self.trans_mat_var.view(1, self.num_labels, self.num_labels, (2*self.gaussian_dim+1)*self.gaussian_dim)

            t_var_size = t_var.size()
            zero_var = torch.zeros((t_var_size[:-1] + (1, )))
            t_vars = torch.split(t_var, [2, 1], dim=-1)
            t_var = torch.cat((t_vars[0], zero_var, t_vars[1]), dim=-1)
            t_var = t_var.view(1, self.num_labels, self.num_labels, 2*self.gaussian_dim, 2*self.gaussian_dim)
            t_var_size = t_var.size()
            t_var = torch.bmm(t_var.view(-1, t_var_size[-1], t_var_size[-2]),
                              t_var.view(-1, t_var_size[-1], t_var_size[-2]).transpose(1, 2)).view(t_var_size)

        # multiply
        # n1 is the big.
        #
        # def general_gaussian_multi(n1_mu, n1_var, n2_mu, n2_var, child):
        #     dim = self.gaussian_dim
        #
        #     def zeta(l, eta, v=None):
        #         # shape reformat
        #         l_size = l.size()
        #
        #         if v is None:
        #             v = self.inverse(l)
        #
        #         dim = l_size[-1]
        #         l = l.view(-1, dim, dim)
        #         eta = eta.view(-1, dim)
        #         v = v.view(-1, dim, dim)
        #
        #         # can optimize
        #         # Alert the scale is log format
        #         scale = -0.5 * (dim * math.log(2 * math.pi) - torch.log(self.determinate(l)) + torch.bmm(
        #             torch.bmm(eta.unsqueeze(1), v), eta.unsqueeze(2)).squeeze())
        #         return scale.view(l_size[:-2])
        #
        #     # n1_var = n1_var.view(n1_var.size()[:-2] + (-1,))
        #
        #     n1_mu = n1_mu.contiguous()
        #     n2_mu = n2_mu.contiguous()
        #     n1_var = n1_var.contiguous()
        #     n2_var = n2_var.contiguous()
        #     # alert matrix multiply
        #     # here V = (Lambda_22 - Lambda_12^T (Lambda_11 + lambda)^-1 Lambda_12)^-1
        #     # lambda is inverse of V
        #     l1 = self.inverse(n1_var)
        #     l1_00, l1_01, l1_10, l1_11 = torch.split(l1.view(l1.size()[:-2] + (-1,)), [dim, dim, dim, dim], dim=-1)
        #     # deal with format
        #     # l1_size [batch, length, pos_num, pos_num, dim, dim]
        #     l1_00 = l1_00.unsqueeze(-1)
        #     l1_01 = l1_01.unsqueeze(-1)
        #     l1_10 = l1_10.unsqueeze(-1)
        #     l1_11 = l1_11.unsqueeze(-1)
        #     l2 = self.inverse(n2_var).contiguous()
        #
        #     # alert format : only support 3d
        #     l1_size = l1.size()
        #     # compressed l1 and n1_mu and eta1
        #     l1 = l1.view(-1, l1_size[-2], l1_size[-1])
        #     n1_mu = n1_mu.view(-1, l1_size[-1], 1)
        #     eta1 = torch.bmm(l1, n1_mu).view(l1_size[:-2] + (2 * dim,))
        #
        #     eta1_0, eta1_1 = torch.split(eta1.view(l1_size[:-1]), [dim, dim], dim=-1)
        #
        #     # compressed l2, n2_mu and eta2
        #     l2_size = l2.size()
        #     n2_mu_size = n2_mu.size()
        #     l2 = l2.view(-1, l2_size[-2], l2_size[-1])
        #     n2_mu = n2_mu.view(-1, l2_size[-2], 1)
        #     eta2 = torch.bmm(l2, n2_mu).view(n2_mu_size)
        #
        #     zeta1 = zeta(l1.contiguous(), eta1.contiguous(), v=n1_var).view(l1_size[:-2])
        #     zeta2 = zeta(l2, eta2, v=n2_var).view(l2_size[:-2])
        #
        #     # extend l1 split part
        #     # l1_part_size = l1_00.size()
        #     # l1_00 = (l1_00 + torch.zeros(l2_size, requires_grad=False).cuda()).view(-1, dim, dim)
        #     # l1_10 = (l1_10 + torch.zeros(l2_size, requires_grad=False).cuda()).view(-1, dim, dim)
        #     # l1_01 = (l1_01 + torch.zeros(l2_size, requires_grad=False).cuda()).view(-1, dim, dim)
        #     # l1_11 = (l1_11 + torch.zeros(l2_size, requires_grad=False).cuda()).view(-1, dim, dim)
        #     # l2 = (l2.view(l2_size) + torch.zeros(l1_part_size, requires_grad=False).cuda())
        #
        #     l1_part_size = l1_00.size()
        #     l1_00 = (l1_00 + torch.zeros(l2_size, requires_grad=False)).view(-1, dim, dim)
        #     l1_10 = (l1_10 + torch.zeros(l2_size, requires_grad=False)).view(-1, dim, dim)
        #     l1_01 = (l1_01 + torch.zeros(l2_size, requires_grad=False)).view(-1, dim, dim)
        #     l1_11 = (l1_11 + torch.zeros(l2_size, requires_grad=False)).view(-1, dim, dim)
        #     l2 = (l2.view(l2_size) + torch.zeros(l1_part_size, requires_grad=False))
        #
        #     var_size = l2.size()
        #     mu_size = var_size[:-1]
        #     scale_size = mu_size[:-1]
        #     l2 = l2.view(-1, dim, dim)
        #
        #     if child:
        #         la_part = torch.bmm(l1_10, self.inverse(l1_11 + l2))
        #         la_new = l1_00 - torch.bmm(la_part, l1_10)
        #         var_new = self.inverse(la_new)
        #
        #         eta1_0 = eta1_0 + torch.zeros_like(eta2)
        #
        #         eta1_0 = eta1_0.view(-1, dim, 1)
        #         eta_new = eta1_0 - torch.bmm(la_part, (eta1_1 + eta2).view(-1, dim, 1))
        #         mu_new = torch.bmm(var_new, eta_new)
        #         zeta_merge = zeta(l1_11 + l2, eta1_1 + eta2)
        #         zeta_new = zeta(la_new, eta_new, var_new)
        #
        #     else:
        #         la_part = torch.bmm(l1_10, self.inverse(l1_00 + l2))
        #         la_new = l1_00 - torch.bmm(la_part, l1_01)
        #         var_new = self.inverse(la_new)
        #
        #         eta1_1 = eta1_1 + torch.zeros_like(eta2)
        #
        #         eta1_1 = eta1_1.view(-1, dim, 1)
        #         eta_new = eta1_1 - torch.bmm(la_part, (eta1_0 + eta2).view(-1, dim, 1))
        #         mu_new = torch.bmm(var_new, eta_new)
        #         zeta_merge = zeta(l1_00 + l2, eta1_0 + eta2)
        #         zeta_new = zeta(la_new, eta_new, var_new)
        #     # alert shape
        #     check_numerics(zeta1)
        #     check_numerics(zeta2)
        #     check_numerics(zeta_merge)
        #     check_numerics(zeta_new)
        #     scale = zeta1 + zeta2 - zeta_merge.view(scale_size) - zeta_new.view(scale_size)
        #     return scale, mu_new.view(mu_size), var_new.view(var_size)

        def general_gaussian_multi(n1_mu, n1_var, n2_mu, n2_var, child):
            dim = 1

            def zeta(l, eta, v=None):
                # shape reformat
                l_size = l.size()

                if v is None:
                    v = self.inverse(l)

                dim = l_size[-1]
                # compress it to 3 dim
                l = l.view(-1, dim, dim)
                eta = eta.view(-1, dim)
                v = v.view(-1, dim, dim)

                # can optimize
                # Alert the scale is log format
                # log(2*pi) = 1.8378
                scale = -0.5 * (dim * 1.837877 - torch.log(self.determinate(l)) + torch.bmm(
                    torch.bmm(eta.unsqueeze(1), v), eta.unsqueeze(2)).squeeze())
                check_numerics(scale)
                return scale.view(l_size[:-2])

            # n1_var = n1_var.view(n1_var.size()[:-2] + (-1,))

            n1_mu = n1_mu.contiguous()
            n2_mu = n2_mu.contiguous()
            n1_var = n1_var.contiguous()
            n2_var = n2_var.contiguous()

            # calculate lambda, eta and zeta
            lambda1 = self.inverse(n1_var)
            lambda2 = self.inverse(n2_var)

            l1_size = lambda1.size()
            l1 = lambda1.view(-1, l1_size[-2], l1_size[-1])
            eta1 = torch.bmm(l1, n1_mu.view(-1, l1_size[-1], 1)).view(l1_size[:-2] + (2 * dim,))
            l1 = l1.view(l1_size)

            l2_size = lambda2.size()
            l2 = lambda2.view(-1, l2_size[-2], l2_size[-1])
            eta2 = torch.bmm(l2, n2_mu.view(-1, l2_size[-1], 1)).view(l2_size[:-2] + (dim,))
            l2 = l2.view(l2_size)

            zeta1 = zeta(l1, eta1, n1_var)
            zeta2 = zeta(l2, eta2, n2_var)

            if not child:
                zero_l2 = torch.zeros_like(l2)
                l2_expand = torch.cat((l2, zero_l2, zero_l2, zero_l2), dim=-2).squeeze(-1)
                l1 = l1.view(l1_size[:-2] + (-1,))
                l_multi = l1 + l2_expand
                l_multi = l_multi.view(l_multi.size()[:-1] + (2 * dim, 2 * dim))

                zero_e2 = torch.zeros_like(eta2)
                eta2_expand = torch.cat((eta2, zero_e2), dim=-1)
                eta1 = eta1.view(l1_size[:-2] + (-1,))
                eta_multi = eta1 + eta2_expand
                eta_multi = eta_multi.view(eta_multi.size()[:-1] + (2 * dim,))
                zeta_multi = zeta(l_multi, eta_multi)
                var_multi = self.inverse(l_multi)
                var_multi_size = var_multi.size()
                mu_multi = torch.bmm(var_multi.view(-1, var_multi_size[-2], var_multi_size[-2]),
                                     eta_multi.view(-1, var_multi_size[-1], 1)).view(var_multi_size[:-1])
                mu_res = torch.split(mu_multi, [dim, dim], dim=-1)[1]
                var_res = torch.split(torch.split(var_multi, [dim, dim], dim=-1)[1], [dim, dim], dim=-2)[1]
            else:
                zero_l2 = torch.zeros_like(l2)
                l2_expand = torch.cat((zero_l2, zero_l2, zero_l2, l2), dim=-2).squeeze(-1)
                l1 = l1.view(l1_size[:-2] + (-1,))
                l_multi = l1 + l2_expand
                l_multi = l_multi.view(l_multi.size()[:-1] + (2 * dim, 2 * dim))

                zero_e2 = torch.zeros_like(eta2)
                eta2_expand = torch.cat((zero_e2, eta2), dim=-1)
                eta1 = eta1.view(l1_size[:-2] + (-1,))
                eta_multi = eta1 + eta2_expand
                eta_multi = eta_multi.view(eta_multi.size()[:-1] + (2 * dim,))
                zeta_multi = zeta(l_multi, eta_multi)
                var_multi = self.inverse(l_multi)
                var_multi_size = var_multi.size()
                mu_multi = torch.bmm(var_multi.view(-1, var_multi_size[-2], var_multi_size[-2]),
                                     eta_multi.view(-1, var_multi_size[-1], 1)).view(var_multi_size[:-1])
                mu_res = torch.split(mu_multi, [dim, dim], dim=-1)[0]
                var_res = torch.split(torch.split(var_multi, [dim, dim], dim=-1)[0], [dim, dim], dim=-2)[0]
            scale = zeta1 + zeta2 - zeta_multi
            return scale, mu_res, var_res

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            var_square_add = n1_var + n2_var
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var + n2_mu * n1_var) / var_square_add

            var = torch.pow(n1_var * n2_var  / var_square_add, 0.5)
            scale = torch.sum(scale, dim=-1)
            return scale, mu, var

        output = []
        mu_p = None
        var_p = None
        for i in range(length):
            if self.bigram:
                if i == 0:
                    scale_c, mu_c, var_c = general_gaussian_multi(t_mu[i], t_var[i], s_mu[i], s_var[i], True)
                    scale_p, mu_p, var_p = general_gaussian_multi(t_mu[i + 1].unsqueeze(1), t_var[i + 1].unsqueeze(1),
                                                                  mu_c.unsqueeze(3), var_c.unsqueeze(3), False)
                else:
                    scale_c, mu_c, var_c = gaussian_multi(mu_p, var_p, s_mu[i], s_var[i])
                    scale_p, mu_p, var_p = general_gaussian_multi(t_mu[i], t_var[i], mu_c, var_c, False)
                scale = scale_c.unsqueeze(3) + scale_p + t_weight[i] + s_weight
            else:
                if i == 0:
                    scale_c, mu_c, var_c = general_gaussian_multi(t_mu, t_var, s_mu[i], s_var[i], True)
                    check_numerics(scale_c)
                    check_numerics(mu_c)
                    check_numerics(var_c)
                    scale_p, mu_p, var_p = general_gaussian_multi(t_mu.unsqueeze(1), t_var.unsqueeze(1),
                                                                  mu_c.unsqueeze(3), var_c.unsqueeze(3), False)
                else:
                    _, mu_p, var_p = self.pruning(scale_p, mu_p, var_p, 1, merge_dims=(1, 2))
                    mu_p = mu_p.unsqueeze(2)
                    var_p = var_p.unsqueeze(2)
                    scale_c, mu_c, var_c = gaussian_multi(mu_p, var_p, s_mu[i], s_var[i].squeeze(-1))
                    scale_p, mu_p, var_p = general_gaussian_multi(t_mu.unsqueeze(1), t_var.unsqueeze(1),
                                                                  mu_c.unsqueeze(3), var_c.unsqueeze(3).unsqueeze(-1), False)
                    if i != length -1:
                        scale_p = scale_p * (mask[:, i+1]).view(batch, 1, 1, 1)
                    else:
                        scale_p = scale_p * (mask[:, i]).view(batch, 1, 1, 1)
                scale = scale_c.unsqueeze(3) + scale_p + t_weight + s_weight[i]
            output.append(scale.view((1,) + scale.size()))
        output = torch.cat(tuple(output), dim=0)

        if mask is not None:
            output = output * mask.transpose(0, 1).view(length, batch, 1, 1, 1)
        return output.transpose(0, 1)

    def loss(self, sents, target, mask):
        batch, length = sents.size()
        energy = self.forward(sents, mask)
        if mask is not None:
            mask_transpose = mask.transpose(0, 1).unsqueeze(2).unsqueeze(3)
        else:
            mask_transpose = None

        energy_transpose = energy.transpose(0, 1)

        target_transpose = target.transpose(0, 1)

        partition = None

        if energy.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, requires_grad=True).cuda()
            holder = torch.zeros(batch).long().cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, requires_grad=True)
            holder = torch.zeros(batch).long()

        for t in range(length):
            # shape = [batch, num_label, num_label, num_label]
            curr_energy = energy_transpose[t]
            if t == 0:
                # partition shape [batch, num_label, num_label]
                partition = curr_energy[:, -1]
            else:
                # shape = [batch, num_label, num_label, num_label]
                partition_new = logsumexp(curr_energy + partition.unsqueeze(3), dim=1)
                if mask_transpose is None:
                    partition = partition_new
                else:
                    mask_t = mask_transpose[t]
                    partition = partition + (partition_new - partition) * mask_t

            if t != length - 1:
                tmp_energy = curr_energy[
                    batch_index, prev_label, target_transpose[t], target_transpose[t + 1]]
            else:
                tmp_energy = curr_energy[
                    batch_index, prev_label, target_transpose[t], holder]
            tgt_energy_new = tmp_energy + tgt_energy
            if mask_transpose is None or t is 0:
                tgt_energy = tgt_energy_new
            else:
                mask_t = mask_transpose[t]
                tgt_energy = tgt_energy + (tgt_energy_new - tgt_energy) * mask_t.squeeze()
            prev_label = target_transpose[t]
        partition = partition.mean(dim=2)
        partition = logsumexp(partition, dim=1)
        loss = partition - tgt_energy
        # if np.min(loss.data.cpu().numpy()) < 0.0:
        #     print("ERROR")
        return loss.mean()

    def decode(self, sents, target, mask, lengths, leading_symbolic=0):
        is_cuda = False
        energy = self.forward(sents, mask).data
        energy_transpose = energy.transpose(0, 1)
        mask = mask.data
        mask_transpose = mask.transpose(0, 1)
        length, batch_size, num_label, _, _ = energy_transpose.size()

        # Forward word and Backward

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask, batch_first=False)

        forward = torch.zeros([length - 1, batch_size, num_label, num_label])# .cuda()
        backward = torch.zeros([length - 1, batch_size, num_label, num_label])# .cuda()
        holder = torch.zeros([1, batch_size, num_label, num_label])# .cuda()
        mask_transpose = mask_transpose.view(length, batch_size, 1, 1)
        for i in range(0, length - 1):
            if i == 0:
                forward[i] = energy_transpose[i, :, -1]
                backward[i] = reverse_energy_transpose[i, :, :, :, 2]
            else:
                forward[i] = logsumexp(forward[i - 1].unsqueeze(3) + energy_transpose[i], dim=1)
                forward[i] = forward[i - 1] + (forward[i] - forward[i - 1]) * mask_transpose[i]
                # backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3) \
                #               * mask_transpose[i].unsqueeze(1).unsqueeze(2)
                backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3)
                backward[i] = backward[i - 1] + (backward[i] - backward[i - 1]) * mask_transpose[i]

        # detect score calculate by forward and backward, should be equal
        # it is right to be here?
        forward_score = logsumexp(forward[-1, :, :, 2], dim=-1)
        backword_score = logsumexp(backward[-1, :, -1, :], dim=-1)
        err = forward_score - backword_score

        backward = reverse_padded_sequence(backward.contiguous(), mask, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        # cnt = logsumexp(forward + backward, dim=-1)
        cnt = (forward + backward) * mask_transpose.view(length, batch_size, 1, 1)
        cnt_transpose = cnt[:, :, leading_symbolic:-1, leading_symbolic:-1]

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
        # viterbi docoding
        pi[0] = cnt[0, :, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].unsqueeze(2)
            pi[t], pointer[t] = torch.max(cnt_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]
        preds = back_pointer.transpose(0, 1) + leading_symbolic
        if target is None:
            return preds, None
        # if lengths is not None:
        #     max_len = lengths.max()
        #     target = target[:, :max_len]
        if mask is None:
            return preds, torch.eq(preds, target.data).float().sum()
        else:
            return preds, (torch.eq(preds, target.data).float() * mask.data).sum()


def natural_data():
    batch_size = 4
    num_epochs = 100
    gaussian_dim = 1
    learning_rate = 1e-1
    momentum = 0.9
    gamma = 0.0
    schedule = 5
    decay_rate = 0.01
    device = torch.device("cuda")

    # train_path = "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn"
    train_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/toy2"
    dev_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/toy2"
    test_path = "/home/ehaschia/Code/NeuroNLP2/data/pos/toy2"

    logger = get_logger("POSCRFTagger")
    # load data

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("/home/ehaschia/Code/NeuroNLP2/data/alphabets/pos_crf/toy2/",
                                                 train_path, data_paths=[dev_path, test_path], normalize_digits=True,
                                                 max_vocabulary_size=50000, embedd_dict=None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    # use_gpu = torch.cuda.is_available()
    use_gpu = False

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet, normalize_digits=False,
                                                   use_gpu=use_gpu)
    # pw_cnt_map, pp_cnt_map = store_cnt(data_train, word_alphabet, pos_alphabet)

    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 use_gpu=use_gpu, volatile=True, normalize_digits=False)
    data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  use_gpu=use_gpu, volatile=True, normalize_digits=False)

    network = lveg(word_alphabet.size(), pos_alphabet.size(), gaussian_dim=gaussian_dim)

    # network = ChainCRF(word_alphabet.size(), pos_alphabet.size())
    # store_gaussians(network, word_alphabet, pos_alphabet, '0', pw_cnt_map, pp_cnt_map, pre, threshold)
    if use_gpu:
        network.to(device)

    lr = learning_rate
    optim = torch.optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    # optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0

    for epoch in range(1, num_epochs + 1):
        print('Epoch %d, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            word, _, labels, _, _, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size,
                                                                                   unk_replace=0.0)

            # max_len = torch.max(lengths) + 1
            # word = word[:, :max_len]
            # labels = labels[:, :max_len]
            # masks = masks[:, :max_len]

            optim.zero_grad()
            loss = network.loss(word, labels, mask=masks).mean()
            loss.backward()
            # store_input(word, labels, masks, batch, epoch)
            # store_data(network, loss, batch, epoch)
            optim.step()

            num_inst = word.size(0)
            train_err += loss.item() * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        # evaluate performance on dev data
        # if epoch % 10 == 1:
        #     store_param(network, epoch, pos_alphabet, word_alphabet)
        # if epoch % 20 == 1:
        #     store_gaussians(network, word_alphabet, pos_alphabet, str(epoch), pw_cnt_map, pp_cnt_map, pre, threshold)

        network.eval()
        dev_corr = 0.0
        dev_total = 0
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch

            # max_len = torch.max(lengths) + 1
            # word = word[:, :max_len]
            # labels = labels[:, :max_len]
            # masks = masks[:, :max_len]

            preds, corr = network.decode(word, mask=masks, target=labels, lengths=None,
                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            corr = (torch.eq(preds, labels.data).float() * masks.data).sum()
            num_tokens = masks.data.sum()
            dev_corr += corr
            dev_total += num_tokens
        print('dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total))

        if dev_correct < dev_corr:
            dev_correct = dev_corr
            best_epoch = epoch

            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_total = 0
            for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                word, char, labels, _, _, masks, lengths = batch

                # max_len = torch.max(lengths) + 1
                # word = word[:, :max_len]
                # labels = labels[:, :max_len]
                # masks = masks[:, :max_len]

                preds, corr = network.decode(word, mask=masks, target=labels, lengths=None,
                                             leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                # corr = (torch.eq(preds, labels.data).float() * masks.data).sum()
                num_tokens = masks.data.sum()
                test_corr += corr
                test_total += num_tokens
            test_correct = test_corr
        print("best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch))
        if test_total != 0:
            print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                test_correct, test_total, test_correct * 100 / test_total, best_epoch))

        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optim = torch.optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
            # optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)


def store_param(network, epoch, pos, word):
    with open("lveg4_" + str(epoch), 'w') as f:
        f.write("pos label \n")
        f.write(str(pos.instance2index))
        f.write("\n\n")

        f.write("word label\n")
        f.write(str(word.instance2index))
        f.write("\n\n")

        f.write("s_mu\n")
        s_mu = np.array2string(network.s_mu_em.weight.data.cpu().numpy(), precision=2, separator=',',
                               suppress_small=True)
        f.write(s_mu)
        f.write("\n\n")
        f.write("s_var\n")
        s_var = np.array2string(network.s_var_em.weight.data.cpu().numpy(), precision=2, separator=',',
                                suppress_small=True)
        f.write(s_var)
        f.write("\n\n")
        f.write("t_p_mu\n")
        t_p_mu = np.array2string(network.trans_p_mu.data.squeeze(2).cpu().numpy(), precision=2, separator=',',
                                 suppress_small=True)
        f.write(t_p_mu)
        f.write("\n\n")
        f.write("t_p_var\n")
        t_p_var = np.array2string(network.trans_p_var.data.squeeze(2).cpu().numpy(), precision=2, separator=',',
                                  suppress_small=True)
        f.write(t_p_var)
        f.write("\n\n")
        f.write("t_c_mu\n")
        t_c_mu = np.array2string(network.trans_c_mu.data.squeeze(2).cpu().numpy(), precision=2, separator=',',
                                 suppress_small=True)
        f.write(t_c_mu)
        f.write("\n\n")
        f.write("t_c_var\n")
        t_c_var = np.array2string(network.trans_c_var.data.squeeze(2).cpu().numpy(), precision=2, separator=',',
                                  suppress_small=True)
        f.write(t_c_var)


def store(file, name, tensor):
    file.write(name)
    file.write(np.array2string(tensor.numpy(), precision=2, separator=',', suppress_small=True))
    file.write("\n\n")


def store_input(word, label, mask, batch, epoch):
    with open("toy2_input_new_v2_e" + str(epoch) + "_b" + str(batch), 'w') as f:
        store(f, "word:\n", word.data.cpu())

        store(f, "label:\n", label.data.cpu())

        store(f, "mask:\n", mask.data.cpu())


def store_data(network, loss, batch, epoch):
    with open("toy2_param_new_v2_e" + str(epoch) + "_b" + str(batch), 'w') as f:
        store(f, "loss:\n", loss.squeeze().data.cpu())

        store(f, "trans_weight:\n", network.trans_weight.squeeze().data.cpu())

        store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data.cpu())

        store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data.cpu())

        store(f, "trans_c_mu:\n", network.trans_c_mu.squeeze().data.cpu())

        store(f, "trans_c_var:\n", network.trans_c_var.squeeze().data.cpu())

        store(f, "s_weight_em:\n", network.s_weight_em.weight.squeeze().data.cpu())

        store(f, "s_mu_em:\n", network.s_mu_em.weight.squeeze().data.cpu())

        store(f, "s_var_em:\n", network.s_var_em.weight.squeeze().data.cpu())

    with open("toy2_grad_new_v2_e" + str(epoch) + "_b" + str(batch), 'w') as f:
        if network.trans_weight.grad is not None:
            store(f, "trans_weight:\n", network.trans_weight.grad.squeeze().data.cpu())
        if network.trans_p_mu.grad is not None:
            store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data.cpu())
        if network.trans_p_var.grad is not None:
            store(f, "trans_p_var:\n", network.trans_p_var.grad.squeeze().data.cpu())
        if network.trans_c_mu.grad is not None:
            store(f, "trans_c_mu:\n", network.trans_c_mu.grad.squeeze().data.cpu())
        if network.trans_c_var.grad is not None:
            store(f, "trans_c_var:\n", network.trans_c_var.grad.squeeze().data.cpu())
        if network.s_weight_em.weight.grad is not None:
            store(f, "s_weight_em:\n", network.s_weight_em.weight.grad.squeeze().data.cpu())
        if network.s_mu_em.weight.grad is not None:
            store(f, "s_mu_em:\n", network.s_mu_em.weight.grad.squeeze().data.cpu())
        if network.s_var_em.weight.grad is not None:
            store(f, "s_var_em:\n", network.s_var_em.weight.grad.squeeze().data.cpu())


def detect_inter(cs_mu, cs_var, t_p_mu, t_p_var):
    with open("inter_param_v3", 'w') as f:
        store(f, "cs_mu:\n", cs_mu.grad.squeeze().cpu().data)

        store(f, "cs_var:\n", cs_var.grad.squeeze().cpu().data)

        store(f, "t_p_mu:\n", t_p_mu.grad.squeeze().cpu().data)

        store(f, "t_p_var:\n", t_p_var.grad.squeeze().cpu().data)


def store_gaussians(network, word_alphabet, label_alphabet, epoch, pw_cnt_map, pp_cnt_map, pre='', threshold=2):
    def store_gaussian(f, str1, str2, weight, mu, var, map):
        key = str1 + '->' + str2
        str1 += '->' + str2 + '\t' + str(weight) + '\t' + str(mu) + '\t' + str(var) + '\t' + str(map[key])
        f.write(str1)
        f.write('\n')

    words = word_alphabet.instances
    word_ins2idx = word_alphabet.instance2index

    labels = label_alphabet.instances
    label_ins2idx = label_alphabet.instance2index

    gaussians_dict = dict()

    with open(pre + "emission_" + epoch, 'w') as f:
        weight = network.s_weight_em.weight.cpu().data.numpy()
        mu = network.s_mu_em.weight.squeeze().cpu().data.numpy()
        var = network.s_var_em.weight.squeeze().cpu().data.numpy()
        for i in labels:
            for j in words:
                if i[0] == '_' or j[0] == '_':
                    continue
                if pw_cnt_map[i + '->' + j] < threshold:
                    continue
                if i not in gaussians_dict:
                    gaussians_dict[i] = dict()
                gaussians_dict[i][j] = tuple([mu[word_ins2idx[j], label_ins2idx[i]],
                                              var[word_ins2idx[j], label_ins2idx[i]]])

                store_gaussian(f, i, j, weight[word_ins2idx[j], label_ins2idx[i]],
                               mu[word_ins2idx[j], label_ins2idx[i]],
                               var[word_ins2idx[j], label_ins2idx[i]], pw_cnt_map)
    gaussian_distance_mat(pre + 'gaussian_mat_' + epoch, gaussians_dict, method_type=0)
    with open(pre + 'transition_' + epoch, 'w') as f:
        weight = network.trans_weight.squeeze().cpu().data.numpy()
        mu_p = network.trans_p_mu.squeeze().cpu().data.numpy()
        var_p = network.trans_p_var.squeeze().cpu().data.numpy()
        mu_c = network.trans_c_mu.squeeze().cpu().data.numpy()
        var_c = network.trans_c_var.squeeze().cpu().data.numpy()

        for i in labels:
            for j in labels:
                if i[0] == '_' or j[0] == '_':
                    continue
                if pp_cnt_map[i + '->' + j] < threshold:
                    continue
                store_gaussian(f, i, j, weight[label_ins2idx[i], label_ins2idx[j]],
                               [mu_p[label_ins2idx[i], label_ins2idx[j]], mu_c[label_ins2idx[i], label_ins2idx[j]]],
                               [var_p[label_ins2idx[i], label_ins2idx[j]], var_c[label_ins2idx[i], label_ins2idx[j]]],
                               pp_cnt_map)


def store_cnt(dataset, word_alphabet, pos_alphabet):
    packs, pack_len = dataset
    words = []
    pos = []
    length = []

    for idx, lens in enumerate(pack_len):

        if lens == 0:
            continue
        else:
            tmp_word, _, tmp_pos, _, _, _, _, tmp_length = packs[idx]
            words += tmp_word.data.cpu().numpy().tolist()
            pos += tmp_pos.data.cpu().numpy().tolist()
            length += tmp_length.data.cpu().numpy().tolist()

    word_list = word_alphabet.instances
    word_ins2idx = word_alphabet.instance2index
    word_idx2ins = dict((v, k) for k, v in word_ins2idx.iteritems())
    pos_list = pos_alphabet.instances
    pos_ins2idx = pos_alphabet.instance2index
    pos_idx2ins = dict((v, k) for k, v in pos_ins2idx.iteritems())
    wp_cnt_map = dict()
    pp_cnt_map = dict()
    for i in word_list:
        for j in pos_list:
            if i[0] == '_' or j[0] == '_':
                continue
            key = j + '->' + i
            wp_cnt_map[key] = 0
    for sen in range(0, len(words)):
        for idx in range(length[sen]):
            word_idx = words[sen][idx]
            pos_idx = pos[sen][idx]

            if word_idx == 0 or pos_idx2ins[pos_idx][0] == '_' or word_idx2ins[word_idx][0] == '_':
                continue
            key = pos_idx2ins[pos_idx] + '->' + word_idx2ins[word_idx]
            wp_cnt_map[key] += 1

    for i in pos_list:
        for j in pos_list:
            if i[0] == '_' or j[0] == '_':
                continue
            key = j + '->' + i
            pp_cnt_map[key] = 0

    for sen in range(0, len(pos)):
        for idx in range(1, length[sen]):
            pre_idx = pos[sen][idx - 1]
            auf_idx = pos[sen][idx]

            if pos_idx2ins[pre_idx] == '_' or pos_idx2ins[auf_idx] == '_':
                continue
            key = pos_idx2ins[pre_idx] + '->' + pos_idx2ins[auf_idx]
            pp_cnt_map[key] += 1

    return wp_cnt_map, pp_cnt_map


def gaussian_distance_mat(filename, gaussians_dict, method_type=0):
    def expection_distance(gaussian1, gaussian2):
        mu1, var1 = gaussian1
        mu2, var2 = gaussian2
        var1_square = math.exp(2.0 * var1)
        var2_square = math.exp(2.0 * var2)
        var_square_add = var1_square + var2_square
        var_log_square_add = math.log(var_square_add)

        scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + math.pow(mu1 - mu2, 2.0) / var_square_add)
        if scale < -50:
            print("Something Error")
        return scale

    def kl_distance(gaussian1, gaussian2):
        # todo
        pass

    method = {0: expection_distance,
              1: kl_distance}

    for k, v in gaussians_dict.iteritems():
        with open(filename + '_' + k, 'w') as f:
            key_list = v.keys()
            words = ''
            for word in key_list:
                words += word + '\t'
            f.write(words)
            f.write('\n')
            for i in key_list:
                for j in key_list:
                    f.write('%.4f' % method[method_type](v[i], v[j]))
                    f.write('\t')
                f.write('\n')


if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    natural_data()
    # main()
