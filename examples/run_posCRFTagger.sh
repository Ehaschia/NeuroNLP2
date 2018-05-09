#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python examples/posCRFTagger.py --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 --num_layers 1 --char_dim 30 --num_filters 30 --tag_space 256 --dropout std --learning_rate 0.01 --decay_rate 0.05 --schedule 5 --gamma 0.0 --dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 --embedding random --embedding_dict "/home/zhaoyp/zlw/pos/NeuroNLP2/data/glove/glove.6B/glove.6B.100d.txt" --train "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn" --dev "/home/zhaoyp/Data/pos/en-ud-dev.conllu_clean_cnn" --test "/home/zhaoyp/Data/pos/en-ud-test.conllu_clean_cnn" --language uden --use-tensorboard --log-dir tensorboard/uden-torch4-v2