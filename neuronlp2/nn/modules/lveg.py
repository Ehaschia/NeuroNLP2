import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from torch.autograd import Variable
from neuronlp2.nlinalg import logsumexp


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
            self.trans_nn_weight = nn.Linear(self.input_size, self.num_labels*self.num_labels)
            self.trans_nn_p_mu = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.gaussian_dim)
            self.trans_nn_c_mu = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.gaussian_dim)
            if not self.spherical:
                self.trans_nn_p_var = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.gaussian_dim)
                self.trans_nn_c_var = nn.Linear(self.input_size, self.num_labels*self.num_labels*self.gaussian_dim)
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
            nn.init.xavier_normal(self.trans_nn_weight.weight)
            nn.init.xavier_normal(self.trans_nn_p_mu.weight)
            nn.init.xavier_normal(self.trans_nn_c_mu.weight)
            nn.init.xavier_normal(self.trans_nn_p_var.weight)
            nn.init.xavier_normal(self.trans_nn_c_var.weight)
            nn.init.constant(self.trans_nn_weight.bias, 0)
            nn.init.constant(self.trans_nn_p_mu.bias, 0)
            nn.init.constant(self.trans_nn_c_mu.bias, 0)
            nn.init.constant(self.trans_nn_p_var.bias, 0)
            nn.init.constant(self.trans_nn_c_var.bias, 0)
        else:
            nn.init.normal(self.trans_mat_weight)
            nn.init.normal(self.trans_mat_p_mu)
            nn.init.normal(self.trans_mat_c_mu)
            nn.init.normal(self.trans_mat_p_var)
            nn.init.normal(self.trans_mat_c_var)

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
        s_weight = self.state_nn_weight(input).unsqueeze(2)
        s_mu = self.state_nn_mu(input).view(batch, length, 1, self.num_labels, self.gaussian_dim)
        if self.spherical:
            s_var = self.state_nn_var(input).view(batch, length, 1, 1, 1)
            s_var = s_var.expand(batch, length, 1, self.num_labels, self.gaussian_dim)
        else:
            s_var = self.state_nn_var(input).view(batch, length, 1, self.num_labels, self.gaussian_dim)

        # t_weight size [batch, length, num_label, num_label]
        # mu and var size [batch, length, num_label, num_label, gaussian_dim]
        # var spherical [batch, length, 1, 1, 1]
        if self.bigram:
            t_weight = self.trans_nn_weight(input).view(batch, length, self.num_labels, self.num_labels)
            t_p_mu = self.trans_nn_p_mu(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
            t_c_mu = self.trans_nn_c_mu(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
            if not self.spherical:
                t_p_var = self.trans_nn_p_var(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
                t_c_var = self.trans_nn_c_var(input).view(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
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
                t_p_var = self.trans_mat_p_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)
                t_c_var = self.trans_mat_c_var.expand(batch, length, self.num_labels, self.num_labels, self.gaussian_dim)

        # Gaussian Multiply:
        # in math S*N` = N1*N2
        # here S is scale, S = -(1/2)*log(2pi + sigma1^2 + sigma2^2) - (1/2)*(mu1 -mu2)^2/(sigma1^2 + sigma2^2) is log form
        # mu of N` is (mu1*sigma2^2 + mu2^sigma1^2)/(sigma1^2 + sigma2^2)
        # var of N` is log(sigma1) + log(sigma2) - (1/2)*log(sigma1^2 + sigma2^2)


        # the tensor now not add weight
        def gaussian_multi(n1_mu, n1_var, n2_mu, n2_var):
            # input tensor has 5 dimension

            n1_var_square = torch.exp(2.0 * n1_var)
            n2_var_square = torch.exp(2.0 * n2_var)
            var_square_add = n1_var_square + n2_var_square
            var_log_square_add = torch.log(var_square_add)

            scale = -0.5 * ((math.pi + var_log_square_add) + torch.pow(n1_mu + n2_mu, 2.0) / var_square_add)
            mu = (n1_mu * n2_var_square + n2_mu * n1_var_square) / var_square_add
            var = n1_var + n2_var - 0.5 * var_log_square_add
            scale = torch.sum(scale, dim=-1)
            return scale, mu, var

        # scale  format [batchm length-1, num_labels, num_labels]
        # tensor format [batch, length-1, num_labels, num_labels, gaussian_dim]
        sp_scale, sp_mu, sp_var = gaussian_multi(s_mu[:, :-1, :, :, :], s_var[:, :-1, :, :, :], t_p_mu[:, :-1, :, :, :],
                                                 t_p_var[:, :-1, :, :, :])
        sp_scale = sp_scale + s_weight[:, :-1, :, :] + t_weight[:, :-1, :, :]
        # scale  format [batchm 1, num_labels, num_labels]
        # tensor format [batch, 1, num_labels, num_labels, gaussian_dim]
        sc_scale, _, _ = gaussian_multi(s_mu[:, -1, :, :, :], s_var[:, -1, :, :, :], t_c_mu[:, -2, :, :, :],
                                        t_c_var[:, -2, :, :, :])
        sc_scale = sc_scale + s_weight[:, -1, :, :] + t_weight[:, -2, :, :]
        # scale  format [batch, length-2, num_labels, num_labels]
        # tensor format [batch, length-2, num_labels, num_labels, gaussian_dim]
        spc_scale, _, _ = gaussian_multi(sp_mu[:, 1:, :, :, :].unsqueeze(4), sp_var[:, 1:, :, :, :].unsqueeze(4),
                                         t_c_mu[:, 1:-1, :, :, :].unsqueeze(2) , t_c_var[:, 1:-1, :, :, :].unsqueeze(2))

        spc_scale = logsumexp(spc_scale, dim=2) + sp_scale[:, 1:, :, :] + t_weight[:, 1:-1, :, :]
        # tensor format [batch, length, num_labels, num_labels]
        output = torch.cat((sp_scale[:, 0, :, :].unsqueeze(1), spc_scale, sc_scale.unsqueeze(1)), dim=1)
        if mask is not None:
            output = output * mask.unsqueeze(2).unsqueeze(3)
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
            tgt_energy = Variable(torch.zeros(batch)).cuda()
        else:
            # shape = [batch]
            batch_index = torch.arange(0, batch).long()
            prev_label = torch.LongTensor(batch).fill_(self.num_labels - 1)
            tgt_energy = Variable(torch.zeros(batch))

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

    def decode(self, input, mask=None, leading_symbolic=0):
        # fixme the decoder is wrong, should implement max-rule parser
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
        energy_transpose = energy_transpose[:, :, leading_symbolic:-1, leading_symbolic:-1]

        length, batch_size, num_label, _ = energy_transpose.size()

        if input.is_cuda:
            batch_index = torch.arange(0, batch_size).long().cuda()
            pi = torch.zeros([length, batch_size, num_label, 1]).cuda()
            pointer = torch.cuda.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.cuda.LongTensor(length, batch_size).zero_()
        else:
            batch_index = torch.arange(0, batch_size).long()
            pi = torch.zeros([length, batch_size, num_label, 1])
            pointer = torch.LongTensor(length, batch_size, num_label).zero_()
            back_pointer = torch.LongTensor(length, batch_size).zero_()

        pi[0] = energy[:, 0, -1, leading_symbolic:-1]
        pointer[0] = -1
        for t in range(1, length):
            pi_prev = pi[t - 1]
            pi[t], pointer[t] = torch.max(energy_transpose[t] + pi_prev, dim=1)

        _, back_pointer[-1] = torch.max(pi[-1], dim=1)
        for t in reversed(range(length - 1)):
            pointer_last = pointer[t + 1]
            back_pointer[t] = pointer_last[batch_index, back_pointer[t + 1]]

        return back_pointer.transpose(0, 1) + leading_symbolic
