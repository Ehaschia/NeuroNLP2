import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
from neuronlp2.nlinalg import logsumexp
import numpy as np
import torch.nn.functional as F
from neuronlp2.nn.utils import sequence_mask, reverse_padded_sequence


class ChainLVeG(nn.Module):
    """
    Implement 1 component mixture gaussian
    """
    def __init__(self, input_size, num_labels, bigram=True, spherical=False,
                 gaussian_dim=1):
        """

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
        """
        super(ChainLVeG, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.spherical = spherical
        self.gaussian_dim = gaussian_dim
        self.min_clip = -5.0
        self.max_clip = 5.0
        # Gaussian for every emission rule
        # weight and var is log form
        self.state_nn_weight = nn.Linear(self.input_size, self.num_labels)
        # every label  is a gaussian_dim dimension gaussian
        self.state_nn_mu = nn.Linear(self.input_size, self.num_labels * self.gaussian_dim)
        if not self.spherical:
            self.state_nn_var = nn.Linear(self.input_size, self.num_labels * self.gaussian_dim)
        else:
            self.state_nn_var = nn.Linear(self.input_size, 1)
        # weight and var is log form
        if self.bigram:
            self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels * self.num_labels)
            self.trans_nn_p_mu = nn.Linear(self.input_size, self.num_labels * self.num_labels * self.gaussian_dim)
            self.trans_nn_c_mu = nn.Linear(self.input_size, self.num_labels * self.num_labels * self.gaussian_dim)
            if not self.spherical:
                self.trans_nn_p_var = nn.Linear(self.input_size, self.num_labels * self.num_labels * self.gaussian_dim)
                self.trans_nn_c_var = nn.Linear(self.input_size, self.num_labels * self.num_labels * self.gaussian_dim)
            else:
                self.trans_nn_p_var = nn.Linear(self.input_size, 1)
                self.trans_nn_c_var = nn.Linear(self.input_size, 1)

            self.register_parameter("trans_mat_weight", None)
            self.register_parameter("trans_mat_p_mu", None)
            self.register_parameter("trans_mat_c_mu", None)
            self.register_parameter("trans_mat_p_var", None)
            self.register_parameter("trans_mat_c_var", None)

        else:
            self.trans_nn_weight = None
            self.trans_nn_p_mu = None
            self.trans_nn_c_mu = None
            self.trans_nn_p_var = None
            self.trans_nn_c_var = None

            self.trans_mat_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels))
            self.trans_mat_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.gaussian_dim))
            self.trans_mat_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.gaussian_dim))
            if not self.spherical:
                self.trans_mat_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.gaussian_dim))
                self.trans_mat_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.gaussian_dim))
            else:
                self.trans_mat_p_var = Parameter(torch.Tensor(1))
                # self.trans_mat_p_var = Parameter(torch.Tensor(1, 1)).view(1, 1, 1)
                self.trans_mat_c_var = Parameter(torch.Tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant(self.state_nn_weight.bias, 0)
        nn.init.constant(self.state_nn_mu.bias, 0)
        nn.init.constant(self.state_nn_var.bias, 0)
        if self.bigram:
            nn.init.xavier_normal_(self.trans_nn_weight.weight)
            nn.init.xavier_normal_(self.trans_nn_p_mu.weight)
            nn.init.xavier_normal_(self.trans_nn_c_mu.weight)
            nn.init.xavier_normal_(self.trans_nn_p_var.weight)
            nn.init.xavier_normal_(self.trans_nn_c_var.weight)
            nn.init.constant_(self.trans_nn_weight.bias, 0)
            nn.init.constant_(self.trans_nn_p_mu.bias, 0)
            nn.init.constant_(self.trans_nn_c_mu.bias, 0)
            nn.init.constant_(self.trans_nn_p_var.bias, 0)
            nn.init.constant_(self.trans_nn_c_var.bias, 0)
        else:
            nn.init.normal_(self.trans_mat_weight)
            nn.init.normal_(self.trans_mat_p_mu)
            nn.init.normal_(self.trans_mat_c_mu)
            nn.init.normal_(self.trans_mat_p_var)
            nn.init.normal_(self.trans_mat_c_var)

    def forward(self, input, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
            the energy tensor with shape = [batch, length, num_label, num_label]

        """

        # check_numerics(input)

        batch, length, _ = input.size()
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
        s_weight = self.state_nn_weight(input)

        s_mu = self.state_nn_mu(input).view(batch, length, self.num_labels, self.gaussian_dim)
        # s_mu = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()

        if self.spherical:
            s_var = self.state_nn_var(input).view(batch, length, 1, 1, 1)
            s_var = s_var.expand(batch, length, self.num_labels, self.gaussian_dim)
        else:
            s_var = self.state_nn_var(input).view(batch, length, self.num_labels, self.gaussian_dim)
        # s_var = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()
        # t_weight size [batch, length, num_label, num_label]
        # mu and var size [batch, length, num_label, num_label, gaussian_dim]
        # var spherical [batch, length, 1, 1, 1]
        if self.bigram:
            t_weight = self.trans_nn_weight(input).view(batch, length, self.num_labels, self.num_labels)
            t_p_mu = self.trans_nn_p_mu(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
            t_c_mu = self.trans_nn_c_mu(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
            if not self.spherical:
                t_p_var = self.trans_nn_p_var(input).view(batch, length, self.num_labels, self.num_labels,
                                                          self.gaussian_dim)
                t_c_var = self.trans_nn_c_var(input).view(batch, length, self.num_labels, self.num_labels,
                                                          self.gaussian_dim)
            else:
                t_p_var = self.trans_nn_p_var(input).view(batch, length, 1, 1, 1)
                t_p_var = t_p_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
                t_c_var = self.trans_nn_c_var(input).view(batch, length, 1, 1, 1)
                t_c_var = t_c_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
        else:
            t_weight = self.trans_mat_weight.unsqueeze(0).unsqueeze(1).expand(batch, length, self.num_labels,
                                                                              self.num_labels)
            t_p_mu = self.trans_mat_p_mu.unsqueeze(0).unsqueeze(1).expand(batch, length, self.num_labels,
                                                                          self.num_labels, self.gaussian_dim)
            t_c_mu = self.trans_mat_c_mu.unsqueeze(0).unsqueeze(1).expand(batch, length, self.num_labels,
                                                                          self.num_labels, self.gaussian_dim)
            if not self.spherical:
                t_p_var = self.trans_mat_p_var.unsqueeze(0).unsqueeze(1).expand(batch, length, self.num_labels,
                                                                                self.num_labels, self.gaussian_dim)
                t_c_var = self.trans_mat_c_var.unsqueeze(0).unsqueeze(1).expand(batch, length, self.num_labels,
                                                                                self.num_labels, self.gaussian_dim)
            else:
                t_p_var = self.trans_mat_p_var.expand(batch, length, self.num_labels, self.num_labels,
                                                      self.gaussian_dim)
                t_c_var = self.trans_mat_c_var.expand(batch, length, self.num_labels, self.num_labels,
                                                      self.gaussian_dim)

        # t_p_mu = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)).cuda()
        # t_p_var = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)).cuda()
        # t_c_mu = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)).cuda()
        # t_c_var = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)).cuda()


        # Gaussian Multiply:
        # in math S*N` = N1*N2
        # here S is scale, S = -(1/2)*log(2pi + sigma1^2 + sigma2^2) - (1/2)*(mu1 -mu2)^2/(sigma1^2 + sigma2^2) is log form
        # mu of N` is (mu1*sigma2^2 + mu2^sigma1^2)/(sigma1^2 + sigma2^2)
        # var of N` is log(sigma1) + log(sigma2) - (1/2)*log(sigma1^2 + sigma2^2)

        # s_mu = F.tanh(s_mu)
        # s_var = F.tanh(s_var)
        # t_p_mu = F.tanh(t_p_mu)
        # t_p_var = F.tanh(t_p_var)
        # t_c_mu = F.tanh(t_c_mu)
        # t_c_var = F.tanh(t_c_var)

        s_mu = torch.clamp(s_mu, min=self.min_clip, max=self.max_clip)
        s_var = torch.clamp(s_var, min=self.min_clip, max=self.max_clip)
        t_p_mu = torch.clamp(t_p_mu, min=self.min_clip, max=self.max_clip)
        t_p_var = torch.clamp(t_p_var, min=self.min_clip, max=self.max_clip)
        t_c_mu = torch.clamp(t_c_mu, min=self.min_clip, max=self.max_clip)
        t_c_var = torch.clamp(t_c_var, min=self.min_clip, max=self.max_clip)
        # check_numerics(s_weight)
        # check_numerics(s_mu)
        # check_numerics(s_var)
        # check_numerics(t_weight)
        # check_numerics(t_p_mu)
        # check_numerics(t_p_var)
        # check_numerics(t_c_mu)
        # check_numerics(t_c_var)

        # the tensor now not add weight
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

    def loss(self, input, target, mask=None):
        """

        Args:
            input: Tensor
                the input tensor with shape = [batch, length, input_size]
            target: Tensor
                the tensor of target labels with shape [batch, length]
            mask:Tensor or None
                the mask tensor with shape = [batch, length]

        Returns: Tensor
                A 1D tensor for minus log likelihood loss
        """
        batch, length, _ = input.size()
        energy = self.forward(input, mask=mask)
        # shape = [length, batch, num_label, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.unsqueeze(2).unsqueeze(3).transpose(0, 1)

        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch)).cuda()
            holder = torch.zeros(batch).long().cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch))
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
                    # fixme alert pos->end_pos
                    partition = partition[:, :, 2]
            if t != length - 1:
                tgt_energy += curr_energy[
                    batch_index, prev_label, target_transpose[t].data, target_transpose[t + 1].data]
                prev_label = target_transpose[t].data
            else:
                tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t].data, holder]
                prev_label = target_transpose[t].data
        # check_numerics(partition)
        # check_numerics(tgt_energy)
        # if torch.min(logsumexp(partition, dim=1) - tgt_energy).data[0] < 0.0:
        #     print("ERROR")
        return logsumexp(partition, dim=1) - tgt_energy

    def decode(self, input, mask=None, leading_symbolic=0):
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
        mask_transpose = mask.data.transpose(0, 1)
        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        # energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _, _ = energy_transpose.size()

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask_transpose, batch_first=False)
        # Forward word and Backward
        forward = torch.zeros([length - 1, batch_size, num_label, num_label]).cuda()
        backward = torch.zeros([length - 1, batch_size, num_label, num_label]).cuda()
        holder = torch.zeros([1, batch_size, num_label, num_label]).cuda()

        for i in range(0, length - 1):
            if i == 0:
                forward[i] = energy_transpose[i, :, -1, :, :]
                backward[i] = reverse_energy_transpose[i, :, :, :, 2]
            else:
                forward[i] = logsumexp(forward[i - 1].unsqueeze(3) + energy_transpose[i], dim=1)
                forward[i] = forward[i - 1] + (forward[i] - forward[i - 1]) * mask_transpose[i].unsqueeze(1).unsqueeze(
                    2)
                # backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3) \
                #               * mask_transpose[i].unsqueeze(1).unsqueeze(2)
                backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3)
                backward[i] = backward[i - 1] + (backward[i] - backward[i - 1]) * mask_transpose[i].unsqueeze(
                    1).unsqueeze(2)

        # detect score calculate by forward and backward, should be equal

        # forward_score = logsumexp(forward[-1, :, :, 2], dim=1)
        # backword_score = logsumexp(backward[-1, :, -1, :], dim=1)
        # err = forward_score - backword_score
        backward = reverse_padded_sequence(backward.contiguous(), mask_transpose, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        cnt = forward + backward
        # cnt = cnt * mask_transpose.unsqueeze(2).unsqueeze(3)
        # cnt_transpose = cnt[:, :, leading_symbolic:-1, leading_symbolic:-1]
        cnt_transpose = cnt[:, :, :-1, :-1]
        length, batch_size, num_label, _ = cnt_transpose.size()

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

        return back_pointer.transpose(0, 1)
