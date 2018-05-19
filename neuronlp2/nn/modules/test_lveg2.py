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
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from neuronlp2.nn.utils import check_numerics
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


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
    def __init__(self, word_size, num_labels, gaussian_dim=1, t_component=1, e_component=1):
        super(lveg, self).__init__()

        self.num_labels = num_labels + 1
        self.gaussian_dim = gaussian_dim
        self.t_comp = t_component
        self.e_comp = e_component
        self.min_clip = -5.0
        self.max_clip = 5.0
        self.trans_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp))
        self.trans_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, gaussian_dim))
        self.trans_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, gaussian_dim))
        self.trans_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, self.t_comp, gaussian_dim))
        self.s_weight_em = Embedding(word_size, self.num_labels * self.e_comp)
        self.s_mu_em = Embedding(word_size, self.num_labels * gaussian_dim * self.e_comp)
        self.s_var_em = Embedding(word_size, self.num_labels * gaussian_dim * self.e_comp)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_weight)
        nn.init.xavier_normal_(self.trans_p_mu)
        nn.init.xavier_normal_(self.trans_p_var)
        nn.init.xavier_normal_(self.trans_c_mu)
        nn.init.xavier_normal_(self.trans_c_var)

    def forward(self, input, mask):

        batch, length = input.size()

        zero_holder = torch.zeros([batch, length]).cuda()

        s_mu = self.s_mu_em(input).view(batch, length, 1, self.num_labels, 1, self.e_comp, self.gaussian_dim)

        s_weight = self.s_weight_em(input).view(batch, length, 1, self.num_labels, 1, self.e_comp)

        s_var = self.s_var_em(input).view(batch, length, 1, self.num_labels, 1, self.e_comp, self.gaussian_dim)

        t_weight = self.trans_weight.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp)

        t_p_mu = self.trans_p_mu.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp, self.gaussian_dim)

        t_p_var = self.trans_p_var.view(1, 1, 1, self.num_labels, self.num_labels, 1, 1, self.t_comp, self.gaussian_dim)

        t_c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.t_comp, 1, self.gaussian_dim)

        t_c_var = self.trans_c_var.view(1, 1, self.num_labels, self.num_labels, self.t_comp, 1, self.gaussian_dim)


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
            # check_numerics(n1_var_square)
            # check_numerics(n2_var_square)
            # check_numerics(var_square_add)
            # check_numerics(var_log_square_add)
            # check_numerics(scale)
            # check_numerics(mu)
            # check_numerics(var)
            return scale, mu, var
        # shape [batch, length, num_labels, num_labels, component, component, gaussian_dim]
        cs_scale, cs_mu, cs_var = gaussian_multi(s_mu, s_var, t_c_mu, t_c_var)

        cs_scale = cs_scale + s_weight
        # check_numerics(cs_scale)
        # shape [batch, length-1, num_labels, num_labels, num_labels, component, component, component, gaussian_dim]
        csp_scale, _, _ = gaussian_multi(cs_mu[:, :-1].unsqueeze(4).unsqueeze(7),
                                         cs_var[:, :-1].unsqueeze(4).unsqueeze(7),
                                         t_p_mu, t_p_var)

        mask1 = torch.split(mask, [1, length-1], dim=1)[1]
        csp_scale = csp_scale * mask1.view(batch, length-1, 1, 1, 1, 1, 1, 1)
        t_weight = t_weight * mask1.view(batch, length-1, 1, 1, 1, 1, 1, 1)
        csp_scale = csp_scale + cs_scale[:, :-1].unsqueeze(4).unsqueeze(7) + t_weight
        # output shape [batch, length, num_labels, num_labels, num_labels, component, component, component]

        output = torch.cat((csp_scale, cs_scale[:, -1].unsqueeze(1).unsqueeze(4).unsqueeze(7).expand(batch, 1, self.num_labels,
                                                                                                     self.num_labels, self.num_labels,
                                                                                                     self.t_comp, self.e_comp,
                                                                                                     self.t_comp)), dim=1)
        if mask is not None:
            output = output * mask.view(batch, length, 1, 1, 1, 1, 1, 1)
        return output

    def loss(self, sents, target, mask):
        # fixme not calculate the toy sample by hand, because is too hard
        batch, length = sents.size()
        energy = self.forward(sents, mask)
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
            tgt_energy_new = logsumexp(logsumexp(tmp_energy + tgt_energy.unsqueeze(2).unsqueeze(3), dim=2), dim=1)
            if mask_transpose is None or t is 0:
                tgt_energy = tgt_energy_new
            else:
                mask_t = mask_transpose[t]
                tgt_energy = tgt_energy + (tgt_energy_new - tgt_energy) * mask_t.squeeze(3).squeeze(2)
            prev_label = target_transpose[t]
        # fixme may here is wrong
        partition = partition.mean(dim=2)
        loss = logsumexp(logsumexp(partition, dim=2), dim=1) - logsumexp(tgt_energy, dim=1)
        return loss.mean()

    def decode(self, sents, target, mask, lengths, leading_symbolic=0):
        is_cuda = True
        energy = self.forward(sents, mask).data
        energy_transpose = energy.transpose(0, 1)
        mask = mask.data
        mask_transpose = mask.transpose(0, 1)
        length, batch_size, num_label, _, _, _, _, _ = energy_transpose.size()

        # Forward word and Backward

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask, batch_first=False)

        forward = torch.zeros([length - 1, batch_size, num_label, num_label, self.t_comp]).cuda()
        backward = torch.zeros([length - 1, batch_size, num_label, num_label, self.t_comp]).cuda()
        holder = torch.zeros([1, batch_size, num_label, num_label, self.t_comp]).cuda()
        mask_transpose = mask_transpose.view(length, batch_size, 1, 1, 1)
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
        # it is right to be here?
        forward_score = logsumexp(logsumexp(forward[-1, :, :, 2], dim=-1), dim=1)
        backword_score = logsumexp(logsumexp(backward[-1, :, -1, :], dim=-1), dim=1)
        err = forward_score - backword_score

        backward = reverse_padded_sequence(backward.contiguous(), mask, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        cnt = logsumexp(forward + backward, dim=-1)
        cnt = cnt * mask_transpose.view(length, batch_size, 1, 1)
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
    batch_size = 16
    num_epochs = 500
    gaussian_dim = 1
    t_comp = 1
    e_comp = 1
    learning_rate = 1e-1
    momentum = 0.9
    gamma = 0.0
    schedule = 5
    decay_rate = 0.05
    device = torch.device("cuda")
    use_tb = True
    if use_tb:
        writer = SummaryWriter(log_dir="/public/sist/home/zhanglw/code/pos/0518/NeuroNLP2/tensorboard/uden/raw-lveg-mask-lr0.1")
    else:
        writer = None

    # train_path = "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn"
    train_path = "/public/sist/home/zhanglw/code/pos/0518/NeuroNLP2/data1/en-ud-train.conllu_clean_cnn"
    dev_path = "/public/sist/home/zhanglw/code/pos/0518/NeuroNLP2/data1/en-ud-dev.conllu_clean_cnn"
    test_path = "/public/sist/home/zhanglw/code/pos/0518/NeuroNLP2/data1/en-ud-test.conllu_clean_cnn"

    logger = get_logger("POSCRFTagger")
    # load data

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/uden/",
                                                 train_path, data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet,
                                                   use_gpu=use_gpu)

    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 use_gpu=use_gpu, volatile=True)
    data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  use_gpu=use_gpu, volatile=True)

    network = lveg(word_alphabet.size(), pos_alphabet.size(), gaussian_dim=gaussian_dim,
                   t_component=t_comp, e_component=e_comp)

    # network = ChainCRF(word_alphabet.size(), pos_alphabet.size())
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
        if use_tb:
            writer.add_scalar("loss/train", train_err / train_total, epoch)

        # evaluate performance on dev data
        # if epoch % 10 == 1:
        #     store_param(network, epoch, pos_alphabet, word_alphabet)
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
        if use_tb:
            writer.add_scalar("acc/dev", dev_corr * 100 / dev_total, epoch)

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
        print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            test_correct, test_total, test_correct * 100 / test_total, best_epoch))

        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optim = torch.optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
            # optim = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=gamma)
    if use_tb:
        writer.close()


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

if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    natural_data()
    # main()

