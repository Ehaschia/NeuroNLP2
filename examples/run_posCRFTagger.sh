#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python examples/posCRFTagger.py --mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 \
 --char_dim 30 --num_filters 30 \
 --learning_rate 0.01 --decay_rate 0.05 --schedule 5 --gamma 0.0 \
 --dropout std --p 0.5 --unk_replace 0.0 --bigram \
 --embedding glove --embedding_dict "data/glove/glove.6B.100d.txt" \
 --train "/home/zhaoyp/Data/pos/en-ud-train.conllu_clean_cnn" --dev "/home/zhaoyp/Data/pos/en-ud-dev.conllu_clean_cnn" --test "/home/zhaoyp/Data/pos/en-ud-test.conllu_clean_cnn"