# -*- coding:utf-8 -*-
"""

A tensorflow implementation for text matching
model in paper MV-LSTM.
author: Bin Zhong
data: 2018-11-12

"""

import tensorflow as tf

class MVLSTM(object):
    def __init__(
      self, max_len_left, max_len_right, vocab_size,
      embedding_size, num_hidden, num_k, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer for both sentences
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1),
                name="W")
            self.embedded_chars_left = tf.nn.embedding_lookup(W, self.input_left)
            self.embedded_chars_right = tf.nn.embedding_lookup(W, self.input_right)

        # Create a bi-directional lstm
        with tf.name_scope('bidirectional_lstm'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)	
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
		
		    # bidirectional_dynamic_rnn 运行网络
            outputs_left, states_left = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars_left, dtype=tf.float32, time_major=False)
            outputs_right, states_left = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.embedded_chars_right, dtype=tf.float32, time_major=False)



        with tf.name_scope('integration'):
            left_seq_encoder = tf.concat(outputs_left, -1, name='left_concat')
            right_seq_encoder = tf.concat(outputs_right, -1, name='right_concat')
            cross = tf.matmul(left_seq_encoder, tf.transpose(right_seq_encoder, [0, 2, 1])) # (N, len, len)
    
        with tf.name_scope('k-max-pooling'):
            cross_resh = tf.reshape(cross, [-1, max_len_left*max_len_right], name='reshape')
            self.k_max_pool = tf.nn.top_k(cross_resh, k=num_k)[0]


        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_k, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")

            self.scores = tf.nn.xw_plus_b(self.k_max_pool, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



