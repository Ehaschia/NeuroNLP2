import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class Extend(nn.Module):
    def __init__(self, a_size, b_size):
        super(Extend, self).__init__()
        self.a = Parameter(torch.Tensor(a_size, a_size, a_size))
        self.b = Parameter(torch.Tensor(1, b_size))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.a)
        nn.init.uniform_(self.b)

    def forward(self, extend):
        a_size, _ = self.a.size()
        _, b_size = self.b.size()
        if extend:
            aa = self.a.expand(a_size, b_size).unsqueeze(1)
            bb = self.b.expand(a_size, b_size).unsqueeze(0)
        else:
            aa = self.a
            bb = self.b
        tmp = aa + bb
        res = tmp / tmp
        return torch.sum(res)


class ExtendP(nn.Module):
    def __init__(self, batch, length, num_label, component, gaussian_dim):
        super(ExtendP, self).__init__()
        self.num_label = num_label
        self.component = component
        self.gaussian_dim = gaussian_dim
        self.batch = batch
        self.length = length

        self.a = Parameter(torch.Tensor(batch, length, num_label, num_label, component, component, gaussian_dim))

        self.b = Parameter(torch.Tensor(num_label, num_label, component, gaussian_dim))

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.uniform_(self.a)
        nn.init.uniform_(self.b)

    def forward(self):
        t_p_var = self.b.unsqueeze(0).unsqueeze(0).expand(self.batch, self.length, self.num_label,
                                                          self.num_label, self.component, self.gaussian_dim)

        tmp = self.a.unsqueeze(4).unsqueeze(7) + t_p_var.unsqueeze(2).unsqueeze(5).unsqueeze(6)
        csp_scale = tmp / tmp
        return torch.sum(csp_scale)


def store(file, name, tensor):
    file.write(name)
    file.write(str(tensor.size()))
    file.write("\n")
    file.write(np.array2string(tensor.numpy(), precision=15, separator=',', suppress_small=True))
    file.write("\n\n")


def store_grad(nerwork, v):
    with open("debug_v" + v, 'w') as f:
        store(f, "a:\n", nerwork.a.squeeze().data.cpu())
        store(f, "b:\n", nerwork.b.squeeze().data.cpu())
        store(f, "a_grad:\n", nerwork.a.grad.squeeze().data.cpu())
        store(f, "b_grad:\n", nerwork.b.grad.squeeze().data.cpu())



torch.random.manual_seed(480)
np.random.seed(480)
a_size = 100
b_size = 100
batch, length, num_label, component, gaussian_dim = 4, 5, 5, 1, 1

use_cuda = True
extend = True
# model = Extend(a_size, b_size)
model = ExtendP(batch, length, num_label, component, gaussian_dim)
if use_cuda:
    model.cuda()

loss = model()

loss.backward()
print(loss.item())
store_grad(model, '1')
