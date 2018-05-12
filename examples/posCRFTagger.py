from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for POS tagging.
"""

import sys

sys.path.append(".")
sys.path.append("..")

import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvCRF, BiVarRecurrentConvCRF, BiRecurrentConvLVeG
from neuronlp2 import utils
import os.path
from tensorboardX import SummaryWriter
# from neuronlp2.nn.modules import lveg
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def store_label(epoch, pred, gold):
    if not os.path.exists(str(epoch) + '_pre'):
        with open(str(epoch) + '_pre', 'w') as f:
            pred = pred.cpu().numpy()
            pred = [','.join(label) for label in pred.astype(str)]
            f.write('\n'.join(pred))
            f.write('\n')
        with open(str(epoch) + '_gold', 'w') as f:
            gold = gold.cpu().numpy()
            gold = [','.join(label) for label in gold.astype(str)]
            f.write('\n'.join(gold))
            f.write('\n')
    else:
        with open(str(epoch) + '_pre', 'a') as f:
            pred = pred.cpu().numpy()
            pred = [','.join(label) for label in pred.astype(str)]
            f.write('\n'.join(pred))
            f.write('\n')
        with open(str(epoch) + '_gold', 'a') as f:
            gold = gold.cpu().numpy()
            gold = [','.join(label) for label in gold.astype(str)]
            f.write('\n'.join(gold))
            f.write('\n')

def calculate_gap(preds, mask, length, begin_labeling):
    if mask.is_cuda:
        mask = mask.cpu().data
    else:
        mask = mask.data()
    if length.is_cuda:
        length = length.cpu().data
    else:
        length = length.data
    if preds.is_cuda:
        preds = preds.cpu().data
    else:
        preds = preds.data
    preds = torch.lt(preds, float(begin_labeling)).type(torch.FloatTensor)
    if length is not None:
        max_len = length.max()
        mask = mask[:, :max_len]
    preds = preds * mask
    return torch.sum(preds).item()

def detect_err(loss, pre_loss):
    if (loss - pre_loss) > 5:
        print("detect error!")
    pre_loss = loss
    return pre_loss

def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['random', 'glove', 'senna', 'sskip', 'polyglot'],
                        help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--lveg', default=False, action='store_true')
    parser.add_argument('--language', type=str, default='wsj')
    parser.add_argument('--spherical', default=False, action='store_true')
    parser.add_argument('--gaussian-dim', type=int, default=1)
    parser.add_argument('--use-tensorboard', default=False, action='store_true')
    parser.add_argument('--log-dir', type=str, default='./tensorboard/')

    args = parser.parse_args()

    logger = get_logger("POSCRFTagger")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    bigram = args.bigram
    use_tb = args.use_tensorboard

    embedding = args.embedding
    embedding_path = args.embedding_dict
    if embedding == 'random':
        embedd_dim = args.dim
        embedd_dict = None
    else:
        embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)
    # embedd_dim = 100
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/" + args.language + '/',
                                                 train_path, data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=embedd_dict)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet,
                                                   use_gpu=use_gpu, symbolic_end=True)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])


    num_data = sum(data_train[1])
    num_labels = pos_alphabet.size()

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 use_gpu=use_gpu, volatile=True, symbolic_end=True)
    data_test = conllx_data.read_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  use_gpu=use_gpu, volatile=True, symbolic_end=True)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if embedding == 'random':
                a_embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
            else:
                if word in embedd_dict:
                    a_embedding = embedd_dict[word]
                elif word.lower() in embedd_dict:
                    a_embedding = embedd_dict[word.lower()]
                else:
                    a_embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                    oov += 1
            table[index, :] = a_embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    initializer = nn.init.xavier_uniform
    if args.dropout == 'std':
        if args.lveg:
            network = BiRecurrentConvLVeG(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters,
                                          window, mode, hidden_size, num_layers, num_labels,
                                          tag_space=tag_space, embedd_word=word_table, bigram=bigram, p_in=p_in,
                                          p_out=p_out, p_rnn=p_rnn, initializer=initializer, gaussian_dim=args.gaussian_dim)
        else:
            network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters,
                                         window, mode, hidden_size, num_layers, num_labels,
                                         tag_space=tag_space, embedd_word=word_table, bigram=bigram, p_in=p_in,
                                         p_out=p_out, p_rnn=p_rnn, initializer=initializer)
    else:
        network = BiVarRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters,
                                        window, mode, hidden_size, num_layers, num_labels,
                                        tag_space=tag_space, embedd_word=word_table, bigram=bigram, p_in=p_in,
                                        p_out=p_out, p_rnn=p_rnn, initializer=initializer)
    network_name = type(network).__name__
    logger.info("Bulid network:" + network_name)

    if use_gpu:
        network.cuda()

    lr = learning_rate
    optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma)
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d, crf=%s" % (
        mode, num_layers, hidden_size, num_filters, tag_space, 'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (
        gamma, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))
    if use_tb:
        writer = SummaryWriter(log_dir=args.log_dir + "/" + network_name)
        writer.add_text('config', str(args))
    else:
        writer = None

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0
    pre_loss = 100
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, args.dropout, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        network.train()
        for batch in range(1, num_batches + 1):
            word, char, labels, _, _, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size,
                                                                                      unk_replace=unk_replace)

            optim.zero_grad()
            loss = network.loss(word, char, labels, mask=masks)
            # loss = network.loss(word, labels, masks)
            loss.backward()
            optim.step()

            num_inst = word.size(0)
            # fixme for torch0.4
            # train_err += loss.data[0] * num_tokens
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
        # debug
        pre_loss = detect_err(train_err / train_total, pre_loss)
        network.eval()
        # evaluate performace on train data
        train_corr = 0.0
        train_total = 0.0
        gap = 0.0
        for batch in conllx_data.iterate_batch_variable(data_train, batch_size):
            word, char, labels, _, _, masks, lengths = batch
            preds, corr = network.decode(word, char, target=labels, mask=masks,
                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            # preds, corr = network.decode(word, target=labels, mask=masks, lengths=lengths,
            #                              leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            num_tokens = masks.data.sum()
            gap += calculate_gap(preds, masks, lengths, conllx_data.NUM_SYMBOLIC_TAGS)
            train_corr += corr
            train_total += num_tokens
        print('train corr: %d, total: %d, acc: %.2f%%, gap_err: %.2f%%' % (train_corr, train_total,
                                                                           train_corr / train_total * 100,
                                                                           gap / train_total * 100))
        if use_tb:
            writer.add_scalar("acc/train", train_corr / train_total * 100, epoch)

        # evaluate loss on dev data
        dev_err = 0.0
        dev_total = 0
        start_time = time.time()
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch

            loss = network.loss(word, char, labels, mask=masks)
            # loss = network.loss(word, labels, mask=masks)

            num_inst = word.size(0)
            # dev_err += loss.data[0] * num_inst
            dev_err += loss.item() * num_inst
            dev_total += num_inst
        print('dev loss: %.4f, time: %.2fs' % (dev_err / dev_total, time.time() - start_time))
        if use_tb:
            writer.add_scalar("loss/dev", dev_err / dev_total, epoch)

        # evaluate performance on dev data
        dev_corr = 0.0
        dev_total = 0
        gap = 0.0
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch
            preds, corr = network.decode(word, char, target=labels, mask=masks,
                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            # preds, corr = network.decode(word, target=labels, mask=masks, lengths=lengths,
            #                              leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
            # if epoch >= 30:
            #     store_label(epoch, preds, labels.data)
            #     exit(0)
            num_tokens = masks.data.sum()
            gap += calculate_gap(preds, masks, lengths, conllx_data.NUM_SYMBOLIC_TAGS)
            dev_corr += corr
            dev_total += num_tokens
        print('dev corr: %d, total: %d, acc: %.2f%%, gap_err: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total,
                                                                         gap * 100 / dev_total))
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
                preds, corr = network.decode(word, char, target=labels, mask=masks,
                                             leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                # preds, corr = network.decode(word, target=labels, mask=masks,lengths=lengths,
                #                              leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                num_tokens = masks.data.sum()
                test_corr += corr
                test_total += num_tokens
            test_correct = test_corr
        if dev_total != 0:
            print("best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch))
        if test_total != 0:
            print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                test_correct, test_total, test_correct * 100 / test_total, best_epoch))

        if epoch % schedule == 0:
            lr = learning_rate / (1.0 + epoch * decay_rate)
            optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
    if use_tb:
        writer.close()

def store(file, name, tensor):
    file.write(name)
    file.write(np.array2string(tensor.numpy(), precision=2, separator=',', suppress_small=True))
    file.write("\n\n")

def store_grad(network, loss, v, is_cuda):
    if is_cuda:
        with open("nnlveg_param_" + v, 'w') as f:
            store(f, "loss:\n", loss.squeeze().data.cpu())
            store(f, "trans_mat_weight:\n", network.lveg.trans_mat_weight.squeeze().data.cpu())
            store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_mu.squeeze().data.cpu())
            store(f, "trans_mat_p_var:\n", network.lveg.trans_mat_p_var.squeeze().data.cpu())
            store(f, "trans_mat_c_mu:\n", network.lveg.trans_mat_c_mu.squeeze().data.cpu())
            store(f, "trans_mat_c_var:\n", network.lveg.trans_mat_c_var.squeeze().data.cpu())
            store(f, "state_nn_weight:\n", network.lveg.state_nn_weight.weight.squeeze().data.cpu())
            store(f, "state_nn_mu:\n", network.lveg.state_nn_mu.weight.squeeze().data.cpu())
            store(f, "state_nn_var:\n", network.lveg.state_nn_var.weight.squeeze().data.cpu())
        with open("nnlveg_grad_" + v, 'w') as f:
            if network.lveg.trans_mat_weight.grad is not None:
                store(f, "trans_mat_weight:\n", network.lveg.trans_mat_weight.grad.squeeze().data.cpu())
            if network.lveg.trans_mat_p_mu.grad is not None:
                store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_mu.grad.squeeze().data.cpu())
            if network.lveg.trans_mat_p_var.grad is not None:
                store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_var.grad.squeeze().data.cpu())
            if network.lveg.trans_mat_c_mu.grad is not None:
                store(f, "trans_mat_c_mu:\n", network.lveg.trans_mat_c_mu.grad.squeeze().data.cpu())
            if network.lveg.trans_mat_c_var.grad is not None:
                store(f, "trans_mat_c_var:\n", network.lveg.trans_mat_c_var.grad.squeeze().data.cpu())
            if network.lveg.state_nn_weight.weight.grad is not None:
                store(f, "state_nn_weight:\n", network.lveg.state_nn_weight.weight.grad.squeeze().data.cpu())
            if network.lveg.state_nn_mu.weight.grad is not None:
                store(f, "state_nn_mu:\n", network.lveg.state_nn_mu.weight.grad.squeeze().data.cpu())
            if network.lveg.state_nn_var.weight.grad is not None:
                store(f, "state_nn_var:\n", network.lveg.state_nn_var.weight.grad.squeeze().data.cpu())
    else:

        with open("nnlveg_param_" + v, 'w') as f:
            store(f, "loss:\n", loss.squeeze().data)

            store(f, "trans_mat_weight:\n", network.lveg.trans_mat_weight.squeeze().data)

            store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_mu.squeeze().data)

            store(f, "trans_mat_p_var:\n", network.lveg.trans_mat_p_var.squeeze().data)

            store(f, "trans_mat_c_mu:\n", network.lveg.trans_mat_c_mu.squeeze().data)

            store(f, "trans_mat_c_var:\n", network.lveg.trans_mat_c_var.squeeze().data)

            store(f, "state_nn_weight:\n", network.lveg.state_nn_weight.squeeze().data)

            store(f, "state_nn_mu:\n", network.lveg.state_nn_mu.squeeze().data)

            store(f, "state_nn_var:\n", network.lveg.s_var_em.squeeze().data)

        with open("nnlveg_grad_" + v, 'w') as f:
            if network.lveg.trans_weight.grad is not None:
                store(f, "trans_mat_weight:\n", network.lveg.trans_mat_weight.grad.squeeze().data)
            if network.lveg.trans_p_mu.grad is not None:
                store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_mu.grad.squeeze().data)
            if network.lveg.trans_p_var.grad is not None:
                store(f, "trans_mat_p_mu:\n", network.lveg.trans_mat_p_var.grad.squeeze().data)
            if network.lveg.trans_c_mu.grad is not None:
                store(f, "trans_mat_c_mu:\n", network.lveg.trans_mat_c_mu.grad.squeeze().data)
            if network.lveg.trans_c_var.grad is not None:
                store(f, "trans_mat_c_var:\n", network.lveg.trans_mat_c_var.grad.squeeze().data)
            if network.lveg.s_weight_em.weight.grad is not None:
                store(f, "state_nn_weight:\n", network.lveg.state_nn_weight.grad.squeeze().data)
            if network.lveg.s_mu_em.weight.grad is not None:
                store(f, "state_nn_mu:\n", network.lveg.state_nn_mu.grad.squeeze().data)
            if network.lveg.s_var_em.weight.grad is not None:
                store(f, "state_nn_var:\n", network.lveg.state_nn_var.grad.squeeze().data)
if __name__ == '__main__':
    torch.random.manual_seed(480)
    np.random.seed(480)
    main()
