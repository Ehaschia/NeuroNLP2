import torch
import torch.nn as nn
import math
from neuronlp2.nlinalg import logsumexp
import numpy as np
from neuronlp2.nn.utils import sequence_mask, reverse_padded_sequence
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import Embedding
from neuronlp2.io import get_logger, conllx_data
import time
import sys
from tensorboardX import SummaryWriter
import torch.nn.functional as F


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
        nn.init.normal(self.trans_matrix)
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


class lveg(nn.Module):
    def __init__(self, num_labels, gaussian_dim, word_size):
        super(lveg, self).__init__()

        self.num_labels = num_labels + 1
        self.gaussian_dim = gaussian_dim

        self.trans_weight = Parameter(torch.Tensor(self.num_labels, self.num_labels))
        self.trans_p_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, gaussian_dim))
        self.trans_p_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, gaussian_dim))
        self.trans_c_mu = Parameter(torch.Tensor(self.num_labels, self.num_labels, gaussian_dim))
        self.trans_c_var = Parameter(torch.Tensor(self.num_labels, self.num_labels, gaussian_dim))
        self.s_weight_em = Embedding(word_size, self.num_labels)
        self.s_mu_em = Embedding(word_size, self.num_labels * gaussian_dim)
        self.s_var_em = Embedding(word_size, self.num_labels * gaussian_dim)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_normal(self.trans_weight)
        nn.init.xavier_normal(self.trans_p_mu)
        nn.init.xavier_normal(self.trans_p_var)
        nn.init.xavier_normal(self.trans_c_mu)
        nn.init.xavier_normal(self.trans_c_var)

    def forward(self, input, mask):

        batch, length = input.size()

        s_mu = self.s_mu_em(input).view(batch, length, self.num_labels, self.gaussian_dim)
        # s_weight = self.s_weight_em(input).view(batch, length, self.num_labels)
        s_weight = Variable(torch.zeros(batch, length, self.num_labels)).cuda()
        s_var = self.s_var_em(input).view(batch, length, self.num_labels, self.gaussian_dim)

        # t_weight = self.trans_weight.view(1, 1, self.num_labels, self.num_labels).expand(batch, length, self.num_labels, self.num_labels)
        t_weight = Variable(torch.zeros(batch, length, self.num_labels, self.num_labels)).cuda()
        t_p_mu = self.trans_p_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length,
                                                                                                        self.num_labels,
                                                                                                        self.num_labels,
                                                                                                        self.gaussian_dim)
        t_p_var = self.trans_p_var.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length,
                                                                                                          self.num_labels,
                                                                                                          self.num_labels,
                                                                                                          self.gaussian_dim)
        t_c_mu = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length,
                                                                                                        self.num_labels,
                                                                                                        self.num_labels,
                                                                                                        self.gaussian_dim)
        t_c_var = self.trans_c_mu.view(1, 1, self.num_labels, self.num_labels, self.gaussian_dim).expand(batch, length,
                                                                                                         self.num_labels,
                                                                                                         self.num_labels,
                                                                                                         self.gaussian_dim)

        s_mu = F.tanh(s_mu)
        s_var = F.tanh(s_var)
        t_p_mu = F.tanh(t_p_mu)
        t_p_var = F.tanh(t_p_var)
        t_c_mu = F.tanh(t_c_mu)
        t_c_var = F.tanh(t_c_var)

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

    def loss(self, sents, target, mask):
        batch, length = sents.size()
        energy = self.forward(sents, mask)
        is_cuda = True
        if mask is not None:
            mask_transpose = mask.transpose(0, 1).unsqueeze(2).unsqueeze(3)
        else:
            mask_transpose = None

        energy_transpose = energy.transpose(0, 1)

        target_transpose = target.transpose(0, 1)

        partition = None

        if is_cuda:
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
                    partition = partition[:, :, 2]
            if t != length - 1:
                tgt_energy += curr_energy[
                    batch_index, prev_label, target_transpose[t], target_transpose[t + 1]]
                prev_label = target_transpose[t]
            else:
                tgt_energy += curr_energy[batch_index, prev_label, target_transpose[t], holder]
                prev_label = target_transpose[t]
        return (logsumexp(partition, dim=1) - tgt_energy).mean()

    def decode(self, sents, target, mask, lengths, leading_symbolic=0):
        is_cuda = True
        energy = self.forward(sents, mask).data
        energy_transpose = energy.transpose(0, 1)
        mask_transpose = mask.transpose(0, 1).data
        length, batch_size, num_label, _, _ = energy_transpose.size()

        # Forward word and Backward

        reverse_energy_transpose = reverse_padded_sequence(energy_transpose, mask_transpose, batch_first=False)

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
        # it is right to be here?
        forward_score = logsumexp(forward[-1, :, :, 2], dim=1)
        backword_score = logsumexp(backward[-1, :, -1, :], dim=1)
        err = forward_score - backword_score

        backward = reverse_padded_sequence(backward.contiguous(), mask_transpose, batch_first=False)
        forward = torch.cat((holder, forward), dim=0)
        backward = torch.cat((backward, holder), dim=0)

        cnt = forward + backward
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


softmax = nn.Softmax()


def generate_data(emission_rules, trans_rules, begin_rule, min_len, len_delta, batch, lveg=False):
    # generate data and padding it
    num_labels, num_words = emission_rules.shape

    label_pad = num_labels
    word_pad = num_words
    begin_pad = label_pad + 1

    labels = []
    words = []
    mask = []

    for i in range(batch):
        length = sampler(softmax(Variable(torch.rand(len_delta))).data.numpy()) + min_len
        mask.append(length)

        label = []
        word = []
        for j in range(length):
            if j == 0:
                a_label = sampler(begin_rule)
            else:
                a_label = sampler(trans_rules[label[j - 1]])
            a_word = sampler(emission_rules[a_label])
            label.append(a_label)
            word.append(a_word)
        if lveg:
            label.append(begin_pad)
            # may should be begin_pad ?
            word.append(word_pad)
        labels.append(label)
        words.append(word)
    # padd data
    if lveg:
        max_len = max(mask) + 1
    else:
        max_len = max(mask)
    for i in range(batch):
        labels[i] = np.pad(labels[i], (0, max_len - len(labels[i])), 'constant', constant_values=(label_pad, label_pad))
        words[i] = np.pad(words[i], (0, max_len - len(words[i])), 'constant', constant_values=(word_pad, word_pad))
    labels = Variable(torch.from_numpy(np.array(labels)))
    words = Variable(torch.from_numpy(np.array(words)))

    mask = torch.from_numpy(np.array(mask))
    mask = Variable(sequence_mask(mask, max_len)).type(torch.FloatTensor)
    return labels, words, mask


def sampler(distrubution):
    rnd = np.random.rand()
    for i in range(len(distrubution)):
        rnd -= distrubution[i]
        if rnd < 0:
            return i


def main():
    num_labels = 5
    words = 10

    trans_rule = Variable(torch.rand(num_labels, num_labels))
    trans_rule = softmax(trans_rule).data.numpy()
    emmsion_rule = Variable(torch.rand(num_labels, words))
    emmsion_rule = softmax(emmsion_rule).data.numpy()
    begin_rule = Variable(torch.rand(num_labels))
    begin_rule = softmax(begin_rule).data.numpy()

    min_len = 5
    len_delta = 5
    batch_size = 16
    gaussian_dim = 1

    epoch = 10000
    train_size = 10
    test_size = 2
    train_set = []
    test_set = []

    for i in range(train_size):
        label_batch, words_batch, mask_batch = generate_data(emmsion_rule, trans_rule, begin_rule, min_len, len_delta,
                                                             batch_size, lveg=True)
        train_set.append((label_batch, words_batch, mask_batch))
    for i in range(test_size):
        label_batch, words_batch, mask_batch = generate_data(emmsion_rule, trans_rule, begin_rule, min_len, len_delta,
                                                             batch_size, lveg=True)
        test_set.append((label_batch, words_batch, mask_batch))
    model = lveg(num_labels + 1, gaussian_dim, words + 1)
    # lveg(num_labels + 2, gaussian_dim, words+1)\
    # ChainCRF(words + 1, num_labels+1)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i in range(epoch):
        train_err = 0
        for gold_label, sents, mask in train_set:
            loss = model.loss(sents, gold_label, mask).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_err += loss.data[0] * batch_size
        print("{} epoch loss {}".format(i, train_err))
        # overfit test
        corr, total = 0, 0
        for gold_label, sents, mask in train_set:
            preds = model.decode(sents, mask, leading_symbolic=0)
            corr += (torch.eq(preds, gold_label.data).float() * mask.data).sum()
            total += mask.data.sum()
        print("      acc is {}".format(corr / total))


def natural_data():
    batch_size = 4
    num_epochs = 200
    gaussian_dim = 3
    learning_rate = 1e-2
    momentum = 0.9
    gamma = 0.0
    schedule = 500
    decay_rate = 0.05
    torch.cuda.device(1)
    use_tb = False
    if use_tb:
        writer = SummaryWriter(log_dir="/home/zhaoyp/zlw/pos/2neuronlp/tensorboard/uden/raw-lveg-lr0.1-dim2")
    else:
        writer = None

    # train_path = "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn"
    train_path = "/home/zhaoyp/Data/pos/toy10"
    dev_path = "/home/zhaoyp/Data/pos/toy10"
    test_path = "/home/zhaoyp/Data/pos/toy10"

    logger = get_logger("POSCRFTagger")
    # load data

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/toy10/",
                                                 train_path, data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet, normalize_digits=False,
                                                   use_gpu=use_gpu, symbolic_end=True)

    num_data = sum(data_train[1])

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 use_gpu=use_gpu, volatile=True, symbolic_end=True,
                                                 normalize_digits=False)
    data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  use_gpu=use_gpu, volatile=True, symbolic_end=True,
                                                  normalize_digits=False)

    network = lveg(pos_alphabet.size(), gaussian_dim, word_alphabet.size())

    # network = ChainCRF(word_alphabet.size(), pos_alphabet.size())
    if use_gpu:
        network.cuda()

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

            optim.zero_grad()
            loss = network.loss(word, labels, mask=masks).mean()
            loss.backward()
            optim.step()

            num_inst = word.size(0)
            train_err += loss.data[0] * num_inst
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
        network.eval()
        dev_corr = 0.0
        dev_total = 0
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch
            preds, corr = network.decode(word, mask=masks, target=labels, lengths=None,
                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            # if epoch >= 30:
            #     store_label(epoch, preds, labels.data)
            #     exit(0)
            # corr = (torch.eq(preds, labels.data).float() * masks.data).sum()
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
    if use_tb:
        writer.close()


def store_param(network, epoch, pos, word):
    with open("lveg_" + str(epoch), 'w') as f:
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


if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    natural_data()
    # main()
