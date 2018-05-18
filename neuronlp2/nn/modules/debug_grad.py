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
            var_square_add = n2_var + n1_var

            #
            scale = n2_mu / var_square_add
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

            # scale = n1_mu + n1_var + n2_mu + n2_var
            return scale

        csp_scale = (t_p_mu.unsqueeze(2).unsqueeze(5).unsqueeze(6) + self.cs_mu.unsqueeze(4).unsqueeze(7)) / \
                    t_p_var.unsqueeze(2).unsqueeze(5).unsqueeze(6)

        return torch.sum(csp_scale)


def store(file, name, tensor):
    file.write(name)
    file.write(str(tensor.size()))
    file.write("\n")
    file.write(np.array2string(tensor.numpy(), precision=10, separator=',', suppress_small=True))
    file.write("\n\n")


def store_extendp(network, v, is_cuda):
    if is_cuda:
        with open("extendp_param_" + v, 'w') as f:
            store(f, "cs_mu:\n", network.cs_mu.squeeze().data.cpu())
            store(f, "cs_var:\n", network.cs_var.squeeze().data.cpu())
            store(f, "trans_p_mu:\n", network.trans_p_mu.squeeze().data.cpu())
            store(f, "trans_p_var:\n", network.trans_p_var.squeeze().data.cpu())
        with open("extendp_grad_" + v, 'w') as f:
            if network.cs_mu.grad is not None:
                store(f, "cs_mu:\n", network.cs_mu.grad.squeeze().data.cpu())
            if network.cs_var.grad is not None:
                store(f, "cs_var:\n", network.cs_var.grad.squeeze().data.cpu())
            if network.trans_p_mu.grad is not None:
                store(f, "trans_p_mu:\n", network.trans_p_mu.grad.squeeze().data.cpu())
            if network.trans_p_var.grad is not None:
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
    batch, length, num_label, component, gaussian_dim = 4, 5, 5, 1, 1
    lr = 0.01
    momentum = 0.9
    gamma = 0.0
    is_cuda = True

    # model = DebugModule(num_label, component, gaussian_dim)
    # model = lveg(num_label, num_label, gaussian_dim=gaussian_dim, component=component)
    # model = lveg2(num_label, num_label, gaussian_dim=gaussian_dim, component=component)
    # model = Minimalist_Mode(batch, length, num_label, component, gaussian_dim)
    model = ExtendP(batch, length, num_label, component, gaussian_dim)
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    sents = torch.randint(0, num_label, (batch, length)).long()

    if is_cuda:
        model.cuda()
        sents.cuda()
    optim.zero_grad()
    loss = model()
    loss.backward()
    store_extendp(model, '4', is_cuda)
    optim.step()
    print(loss.item())


if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    main()
    # natural_data()
