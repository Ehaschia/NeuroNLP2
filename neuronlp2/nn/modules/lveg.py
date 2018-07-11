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
                 t_comp=1, e_comp=1, gaussian_dim=1, clip=1):
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
        self.min_clip = -clip
        self.max_clip = clip
        self.t_comp = t_comp
        self.e_comp = e_comp
        # Gaussian for every emission rule
        # weight and var is log form
        self.state_nn_weight = nn.Linear(self.input_size, self.num_labels*self.e_comp)
        # every label  is a gaussian_dim dimension gaussian
        self.state_nn_mu = nn.Linear(self.input_size, self.num_labels*self.gaussian_dim*self.e_comp)
        if not self.spherical:
            self.state_nn_var = nn.Linear(self.input_size, self.num_labels*self.gaussian_dim*self.e_comp)
        else:
            self.state_nn_var = nn.Linear(self.input_size, 1)
        # weight and var is log form
        if self.bigram:
            self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp)
            self.trans_nn_p_mu = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*self.gaussian_dim)
            self.trans_nn_c_mu = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*self.gaussian_dim)
            if not self.spherical:
                self.trans_nn_p_var = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*self.gaussian_dim)
                self.trans_nn_c_var = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*self.gaussian_dim)
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

            self.trans_mat_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp))
            self.trans_mat_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, self.gaussian_dim))
            self.trans_mat_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, self.gaussian_dim))
            if not self.spherical:
                self.trans_mat_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, self.gaussian_dim))
                self.trans_mat_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, self.gaussian_dim))
            else:
                self.trans_mat_p_var = Parameter(torch.Tensor(1))
                # self.trans_mat_p_var = Parameter(torch.Tensor(1, 1)).view(1, 1, 1)
                self.trans_mat_c_var = Parameter(torch.Tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.state_nn_weight.bias, 0)
        nn.init.constant_(self.state_nn_mu.bias, 0)
        nn.init.constant_(self.state_nn_var.bias, 0)
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

        s_weight = self.state_nn_weight(input).view(batch, length, 1, self.num_labels, 1, self.e_comp)

        s_mu = self.state_nn_mu(input).view(batch, length, 1, self.num_labels, 1, self.e_comp, self.gaussian_dim)
        # s_mu = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()

        if self.spherical:
            s_var = self.state_nn_var(input).view(batch, length, 1, 1, 1, 1, 1)
            # s_var = s_var.expand(batch, length, self.num_labels, self.gaussian_dim)
        else:
            s_var = self.state_nn_var(input).view(batch, length, 1, self.num_labels, 1, self.e_comp, self.gaussian_dim)
        # s_var = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()
        # t_weight size [batch, length, num_label, num_label]
        # mu and var size [batch, length, num_label, num_label, gaussian_dim]
        # var spherical [batch, length, 1, 1, 1]
        if self.bigram:
            t_weight = self.trans_nn_weight(input).view(batch, length, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp)
            t_p_mu = self.trans_nn_p_mu(input).view(batch, length, 1, self.num_labels, self.num_labels, 1, 1,
                                                    self.t_comp, self.gaussian_dim)
            t_c_mu = self.trans_nn_c_mu(input).view(batch, length, self.num_labels, self.num_labels, self.t_comp,
                                                    1, self.gaussian_dim)
            if not self.spherical:
                t_p_var = self.trans_nn_p_var(input).view(batch, length, 1, self.num_labels, self.num_labels, 1, 1,
                                                          self.t_comp, self.gaussian_dim)
                t_c_var = self.trans_nn_c_var(input).view(batch, length, self.num_labels, self.num_labels, self.t_comp,
                                                          1, self.gaussian_dim)
            else:
                t_p_var = self.trans_nn_p_var(input).view(batch, length, 1, 1, 1, 1, 1, 1, 1)
                # t_p_var = t_p_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
                t_c_var = self.trans_nn_c_var(input).view(batch, length, 1, 1, 1, 1, 1, 1, 1)
                # t_c_var = t_c_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
        else:
            t_weight = self.trans_mat_weight.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp)
            t_p_mu = self.trans_mat_p_mu.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp, self.gaussian_dim)
            t_c_mu = self.trans_mat_c_mu.view(1, 1, self.num_labels, self.num_labels, self.t_comp, 1, self.gaussian_dim)
            if not self.spherical:
                t_p_var = self.trans_mat_p_var.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp, self.gaussian_dim)
                t_c_var = self.trans_mat_c_var.view(1, 1, self.num_labels, self.num_labels, self.t_comp, 1, self.gaussian_dim)
            else:
                t_p_var = self.trans_mat_p_var.view(1, 1, 1, 1, 1, 1, 1, 1, 1)
                t_c_var = self.trans_mat_c_var.view(1, 1, 1, 1, 1, 1, 1, 1)



        # Gaussian Multiply:
        # in math S*N` = N1*N2
        # here S is scale, S = -(1/2)*log(2pi + sigma1^2 + sigma2^2) - (1/2)*(mu1 -mu2)^2/(sigma1^2 + sigma2^2) is log form
        # mu of N` is (mu1*sigma2^2 + mu2^sigma1^2)/(sigma1^2 + sigma2^2)
        # var of N` is log(sigma1) + log(sigma2) - (1/2)*log(sigma1^2 + sigma2^2)

        # s_mu_numpy = s_mu.data.squeeze().cpu().numpy()
        # s_var_numpy = s_var.data.squeeze().cpu().numpy()
        # t_p_mu_numpy = t_p_mu.data.squeeze().cpu().numpy()
        # t_p_var_numpy = t_p_var.data.squeeze().cpu().numpy()
        # t_c_mu_numpy = t_c_mu.data.squeeze().cpu().numpy()
        # t_c_var_numpy = t_c_var.data.squeeze().cpu().numpy()

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

        # self.save_parameter(s_weight, s_mu, s_var, t_weight, t_p_mu, t_p_var, t_c_mu, t_c_var, input)

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

        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu, s_var, t_c_mu, t_c_var)

        cs_scale = cs_scale + s_weight
        mask1 = torch.split(mask, [1, length-1], dim=1)[1].view(batch, length-1, 1, 1, 1, 1, 1, 1)
        if self.bigram:
            csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                             cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                             t_p_mu[:, 1:], t_p_var[:, 1:])
            csp_scale = csp_scale * mask1
            t_weight_part = t_weight[:, 1:] * mask1
            csp_scale = csp_scale + cs_scale[:, :-1].unsqueeze(4).unsqueeze(7) + t_weight_part
        else:
            csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                             cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                             t_p_mu, t_p_var)
            csp_scale = csp_scale * mask1
            t_weight_part = t_weight * mask1
            csp_scale = csp_scale + cs_scale[:, :-1].unsqueeze(4).unsqueeze(7) + t_weight_part

        # fixme is this expand ok?
        output = torch.cat((csp_scale, cs_scale[:, -1].unsqueeze(1).unsqueeze(4).unsqueeze(7).expand(batch, 1, self.num_labels,
                                                                                                     self.num_labels, self.num_labels,
                                                                                                     self.t_comp, self.e_comp,
                                                                                                     self.t_comp)), dim=1)
        if mask is not None:
            output = output * mask.view(batch, length, 1, 1, 1, 1, 1, 1)
        # s_weight.retain_grad()
        # s_mu.retain_grad()
        # s_var.retain_grad()
        # t_weight.retain_grad()
        # t_p_mu.retain_grad()
        # t_p_var.retain_grad()
        # t_c_var.retain_grad()
        # t_c_mu.retain_grad()
        # cs_scale.retain_grad()
        # csp_scale.retain_grad()

        return output # , (s_weight, s_mu, s_var, t_weight, t_p_mu, t_p_var, t_c_mu, t_c_var, cs_scale, cs_mu, cs_var,
                      #   csp_scale, s_mu_numpy, s_var_numpy, t_p_mu_numpy, t_p_var_numpy, t_c_mu_numpy, t_c_var_numpy)

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
        # energy.retain_grad()
        # shape = [length, batch, num_label, num_label, num_label]
        energy_transpose = energy.transpose(0, 1)
        # shape = [length, batch]
        target_transpose = target.transpose(0, 1)
        # shape = [length, batch, 1]
        mask_transpose = None
        if mask is not None:
            mask_transpose = mask.transpose(0, 1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # shape = [batch, num_label]
        partition = None

        if input.is_cuda:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long().cuda()
            prev_label = torch.cuda.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, self.t_comp, requires_grad=True).cuda()
            holder = torch.zeros(batch).long().cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = torch.zeros(batch, self.t_comp, requires_grad=True)
            holder = torch.zeros(batch).long()

        for t in range(length):
            # shape = [batch, num_label, num_label, num_label]
            curr_energy = energy_transpose[t]
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
        # alert what's this?
        partition = partition.mean(dim=2)
        loss = (logsumexp(logsumexp(partition, dim=2), dim=1) - logsumexp(tgt_energy, dim=1)).mean()
        return loss #, (energy, to_store)

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
        mask = mask.data
        mask_transpose = mask.transpose(0, 1)
        # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
        # also remove the first #symbolic rows and columns.
        # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - #symbolic - 1.
        # energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _, _, _, _, _ = energy_transpose.size()

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask, batch_first=False)

        forward = torch.zeros([length - 1, batch_size, num_label, num_label, self.t_comp]).cuda()
        backward = torch.zeros([length - 1, batch_size, num_label, num_label, self.t_comp]).cuda()
        holder = torch.zeros([1, batch_size, num_label, num_label, self.t_comp]).cuda()
        mask_transpose = mask_transpose.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # fixme version 2  remove leading_symbolic before expect_count
        for i in range(0, length - 1):
            if i == 0:
                forward[i] = logsumexp(logsumexp(energy_transpose[i, :, -1], dim=4), dim=3)
                backward[i] = logsumexp(logsumexp(reverse_energy_transpose[i, :, :, :, 2], dim=5), dim=4)
            else:
                forward[i] = logsumexp(logsumexp(logsumexp(forward[i - 1].unsqueeze(3).unsqueeze(5).unsqueeze(6) + energy_transpose[i], dim=5), dim=4), dim=1)
                forward[i] = forward[i - 1] + (forward[i] - forward[i - 1]) * mask_transpose[i]
                # backward[i] = logsumexp(backward[i - 1].unsqueeze(1) + reverse_energy_transpose[i], dim=3) \
                #               * mask_transpose[i].unsqueeze(1).unsqueeze(2)
                backward[i] = logsumexp(logsumexp(logsumexp(backward[i - 1].unsqueeze(1).unsqueeze(4).unsqueeze(5) + reverse_energy_transpose[i], dim=6), dim=5), dim=3)
                backward[i] = backward[i - 1] + (backward[i] - backward[i - 1]) * mask_transpose[i]

        # detect score calculate by forward and backward, should be equal

        # forward_score = logsumexp(forward[-1, :, :, 2], dim=1)
        # backword_score = logsumexp(backward[-1, :, -1, :], dim=1)
        # err = forward_score - backword_score
        backward = reverse_padded_sequence(backward.contiguous(), mask, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        # cnt = forward + backward
        # cnt = cnt * mask_transpose.unsqueeze(2).unsqueeze(3)
        # cnt_transpose = cnt[:, :, leading_symbolic:-1, leading_symbolic:-1]
        cnt = logsumexp(forward + backward, dim=-1)
        cnt_transpose = cnt[:, :, leading_symbolic:-1, leading_symbolic:-1]
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
        pi[0] = cnt[0, :, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1].unsqueeze(2)
            pi[t], pointer[t] = torch.max(cnt_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic

    def save_parameter(self, s_weight, s_mu, s_var, t_weight, t_p_mu, t_p_var, t_c_mu, t_c_var, input):

        def precision(tensor):
            return np.array2string(tensor.numpy(), precision=2, separator=',', suppress_small=True, threshold=int(1e8))

        with open("/home/zhaoyp/zlw/pos/2neuronlp/s_weight", 'w') as f:
            f.write(precision(s_weight.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/s_mu", 'w') as f:
            f.write(precision(s_mu.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/s_var", 'w') as f:
            f.write(precision(s_var.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/t_weight", 'w') as f:
            f.write(precision(t_weight.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/t_p_mu", 'w') as f:
            f.write(precision(t_p_mu.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/t_p_var", 'w') as f:
            f.write(precision(t_p_var.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/t_c_mu", 'w') as f:
            f.write(precision(t_c_mu.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/t_c_var", 'w') as f:
            f.write(precision(t_c_var.squeeze().data.cpu()))
        with open("/home/zhaoyp/zlw/pos/2neuronlp/input", 'w') as f:
            f.write(precision(input.squeeze().data.cpu()))

class GeneralLVeG(nn.Module):
    def __init__(self, input_size, num_labels, bigram=True, spherical=False,
                 t_comp=1, e_comp=1, gaussian_dim=1, clip=1):
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
        super(GeneralLVeG, self).__init__()
        self.input_size = input_size
        self.num_labels = num_labels + 1
        self.pad_label_id = num_labels
        self.bigram = bigram
        self.spherical = spherical
        self.gaussian_dim = gaussian_dim
        self.min_clip = -clip
        self.max_clip = clip
        self.t_comp = t_comp
        self.e_comp = e_comp
        # Gaussian for every emission rule
        # weight is log form
        # alert because need to calculate the inverse of variance, so variance should be in normal form.
        self.state_nn_weight = nn.Linear(self.input_size, self.num_labels*self.e_comp)
        # every label  is a gaussian_dim dimension gaussian
        self.state_nn_mu = nn.Linear(self.input_size, self.num_labels*self.gaussian_dim*self.gaussian_dim*self.e_comp)
        # fixme rewrite var, let it be symmetric positive definite matrix
        self.state_nn_var = nn.Linear(self.input_size, self.num_labels*self.gaussian_dim*self.gaussian_dim*self.e_comp)
        # weight and var is log form
        if self.bigram:
            self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp)
            self.trans_nn_mu = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*2*self.gaussian_dim*2*self.gaussian_dim)
            # fixme rewrite var, let it be symmetric positive definite matrix
            self.trans_nn_var = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.t_comp*2*self.gaussian_dim*2*self.gaussian_dim)

            self.register_parameter("trans_mat_weight", None)
            self.register_parameter("trans_mat_mu", None)
            self.register_parameter("trans_mat_var", None)

        else:
            self.trans_nn_weight = None
            self.trans_nn_p_mu = None
            self.trans_nn_c_mu = None
            self.trans_nn_p_var = None
            self.trans_nn_c_var = None

            self.trans_mat_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp))
            self.trans_mat_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, self.gaussian_dim, self.gaussian_dim))
            # fixme rewrite var, let it be symmetric positive definite matrix
            self.trans_mat_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, 2*self.gaussian_dim, 2*gaussian_dim))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.state_nn_weight.bias, 0)
        nn.init.constant_(self.state_nn_mu.bias, 0)
        nn.init.constant_(self.state_nn_var.bias, 0)
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
        # alert the inverse is for normal format
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
            return input
            # return -1.0 * input

        input = input.view(-1, size[-2], size[-1])
        if size[-1] == 2:
            # http://www.mathcentre.ac.uk/resources/uploaded/sigma-matrices7-2009-1.pdf
            det = input[:, 0, 0] * input[:, 1, 1] - input[:, 0, 1] * input[:, 1, 0]
            return det.view(size[:-2])
        else:
            raise NotImplementedError

    def forward(self, input, mask=None):

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

        s_weight = self.state_nn_weight(input).view(batch, length, self.num_labels, self.e_comp).transpose(0, 1)

        s_mu = self.state_nn_mu(input).view(batch, length, self.num_labels, self.e_comp, self.gaussian_dim).transpose(0, 1)
        # s_mu = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()
        s_var = self.state_nn_var(input).view(batch, length, self.num_labels, self.e_comp, self.gaussian_dim, self.gaussian_dim).transpose(0, 1)

        # s_var = Variable(torch.zeros(batch, length, self.num_labels, self.gaussian_dim)).cuda()
        # t_weight size [batch, length, num_label, num_label]
        # mu and var size [batch, length, num_label, num_label, gaussian_dim]

        if self.bigram:
            t_weight = self.trans_nn_weight(input).view(batch, length, self.num_labels, self.num_labels, self.t_comp).transpose(0, 1)
            t_mu = self.trans_nn_p_mu(input).view(batch, length, self.num_labels, self.num_labels,
                                                  self.t_comp, 2*self.gaussian_dim).transpose(0, 1)
            t_var = self.trans_nn_p_var(input).view(batch, length, self.num_labels, self.num_labels,
                                                    self.t_comp, 2*self.gaussian_dim, 2*self.gaussian_dim).transpose(0, 1)
        else:
            t_weight = self.trans_mat_weight.view(1, self.num_labels, self.num_labels, self.t_comp)
            t_mu = self.trans_mat_p_mu.view(1, self.num_labels, self.num_labels, self.t_comp, 2*self.gaussian_dim)
            t_var = self.trans_mat_p_var.view(1, self.num_labels, self.num_labels, self.t_comp, 2*self.gaussian_dim, 2*self.gaussian_dim)

        # multiply
        # n1 is the big.
        def general_gaussian_multi(n1_mu, n1_var, n2_mu, n2_var, child):
            # var size: last 2 dim should be equal
            def zeta(l, eta, v=None):
                if v is None:
                    v = self.inverse(l)
                dim = l.size()[-1]
                bmm_size = l.size()
                # can optimize
                # Alert the scale is log format
                scale = -0.5 * (dim * math.log(2 * math.pi) - torch.log(self.determinate(l)) + torch.bmm(
                    torch.bmm(eta.view(-1, 1, dim), v.view(-1, dim, dim)), eta.view(-1, dim, 1)).view(bmm_size[:-2]))
                return scale

            # n1_var = n1_var.view(n1_var.size()[:-2] + (-1,))
            dim = 1

            # alert matrix multiply
            # here V = (Lambda_22 - Lambda_12^T (Lambda_11 + lambda)^-1 Lambda_12)^-1
            # lambda is inverse of V
            l1 = self.inverse(n1_var)
            l1_00, l1_01, l1_10, l1_11 = torch.split(l1.view(l1.size()[:-2] + (-1,)), [dim, dim, dim, dim], dim=-1)
            l1_00 = l1_00.view(l1_00.size() + (1, 1))
            l1_01 = l1_01.view(l1_01.size() + (1, 1))
            l1_10 = l1_10.view(l1_10.size() + (1, 1))
            l1_11 = l1_11.view(l1_11.size() + (1, 1))
            l2 = self.inverse(n2_var)
            # alert format : only support 3d
            bmm_size = l1.size()
            eta1 = torch.bmm(l1.view(-1, bmm_size[-2], bmm_size[-1]), n1_mu.view(-1, bmm_size[-1], 1)).squeeze(-1)
            eta1_0, eta1_1 = torch.split(eta1.view(bmm_size[:-1]), [dim, dim], dim=-1)
            bmm_size = l2.size()
            eta2 = torch.bmm(l2.view(-1, bmm_size[-2], bmm_size[-1]), n2_mu.view(-1, bmm_size[-1], 1)).squeeze(-1)
            zeta1 = zeta(l1, eta1, v=n1_var)
            zeta2 = zeta(l2, eta2, v=n2_var)

            if child:
                mu_size = n2_mu.size()
                var_size = n2_var.size()
                la_part = torch.bmm(l1_10, self.inverse(l1_11 + l2))
                la_new = l1_00 - torch.bmm(la_part, l1_01).view(var_size)
                var_new = self.inverse(la_new)
                # here eta format is compress
                eta_new = eta1_0 - torch.bmm(la_part, (eta1_1 + eta2).unsqueeze(-1))
                mu_new = torch.bmm(var_new, eta_new).view(mu_size)
                zeta_merge = zeta(l1_11 + l2, eta1_1 + eta2)
                zeta_new = zeta(la_new, eta_new, var_new)

            else:
                mu_size = n2_mu.size()
                var_size = n2_var.size()
                la_part = torch.bmm(l1_10, self.inverse(l1_00 + l2))
                la_new = l1_11 - torch.bmm(la_part, l1_01).view(var_size)
                var_new = self.inverse(la_new)
                # here eta format is compress
                eta_new = eta1_1 - torch.bmm(la_part, (eta1_0 + eta2).unsqueeze(-1))
                mu_new = torch.bmm(var_new, eta_new).view(mu_size)
                zeta_merge = zeta(l1_00 + l2, eta1_0 + eta2)
                zeta_new = zeta(la_new, eta_new, var_new)
            # alert shape
            scale = zeta1 + zeta2 - zeta_merge - zeta_new
            return scale, mu_new, var_new

        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            var_square_add = n1_var + n2_var
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * (math.log(math.pi * 2) + var_log_square_add + torch.pow(n1_mu - n2_mu, 2.0) / var_square_add)

            mu = (n1_mu * n2_var + n2_mu * n1_var) / var_square_add

            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            return scale, mu, var

        output = []
        mu_p = None
        var_p = None
        for i in range(length):
            if i == 0:
                scale_c, mu_c, var_c = general_gaussian_multi(t_mu[i], t_var[i], s_mu[i], s_var[i], True)
                scale_p, mu_p, var_p = general_gaussian_multi(t_mu[i+1], t_var[i+1], mu_c, var_c, False)
            else:
                scale_c, mu_c, var_c = gaussian_multi(mu_p, var_p, s_mu[i], s_var[i])
                scale_p, mu_p, var_p = general_gaussian_multi(t_mu[i], t_var[i], mu_c, var_c, False)
            scale = scale_c + scale_p
            if self.bigram:
                scale += t_weight[i] + s_weight
            else:
                scale += t_weight + s_weight
            output.append(scale.view((1, ) + scale.size()))
        output = torch.cat(tuple(output), dim=0)

        if mask is not None:
            output = output * mask
        return output