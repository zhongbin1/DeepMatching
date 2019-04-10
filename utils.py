# -*- coding:utf-8 -*-

from models.L2R import L2R
from models.ESIM import ESIM
from models.MatchPyramid import MatchPyramid
from models.MVLSTM import MVLSTM
from models.Self_Attention import Self_Attention
from models.DSA import DSA

def get_model(FLAGS, vocab):
    model_name = FLAGS.model_name
    model = None
    if model_name == 'l2r':
        model = L2R(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_hidden=FLAGS.num_hidden,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
    elif model_name == 'mvlstm':
        model = MVLSTM(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            num_k=FLAGS.num_k,
            num_hidden=FLAGS.num_hidden,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
    elif model_name == 'matchpyramid':
        model = MatchPyramid(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            filter_size=3,
            num_filters=FLAGS.num_filters,
            num_hidden=FLAGS.num_hidden,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
    elif model_name == 'self_attention':
        model = Self_Attention(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            num_hidden=300,
            num_da = 350,
            num_r = 10,
            num_mlp_hidden = 128,
            l2_reg_lambda=1.0)
    elif model_name == 'esim':
        model = ESIM(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            rnn_size=300,
            num_hidden=128,
            l2_reg_lambda=0.0)
    elif model_name == 'dsa':
        model = DSA(
            max_len_left=FLAGS.max_len_left,
            max_len_right=FLAGS.max_len_right,
            vocab_size=len(vocab),
            embedding_size=FLAGS.embedding_dim,
            d_1=150,
            d_l=75,
            k_1=3,
            k_2=5, num_layers=4, d_c=300, num_attentions=8, d_o=300, num_iter=2,
            num_hidden=FLAGS.num_hidden,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
    else:
        raise NotImplementedError
    return model
