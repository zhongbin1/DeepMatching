# -*- coding:utf-8 -*-

import tensorflow as tf

def get_mask_attention(inputs, lengths, max_length):
    sequence_mask = tf.sequence_mask(lengths, max_length, dtype=tf.float32)
    sequence_mask = tf.expand_dims(sequence_mask, axis=1)  # [batch, 1, max_len]
    A = inputs * sequence_mask + (sequence_mask - 1) * 1e9
    A = tf.nn.softmax(A, axis=-1)
    return A

class Self_Attention(object):
    def __init__(
      self, max_len_left, max_len_right, vocab_size,
      embedding_size, num_hidden, num_da, num_r, num_mlp_hidden, l2_reg_lambda=0.0):
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.layers.l2_regularizer(l2_reg_lambda)

        self.input_left = tf.placeholder(shape=[None, max_len_left], dtype=tf.int32, name='input_x_left')
        self.input_right = tf.placeholder(shape=[None, max_len_right], dtype=tf.int32, name='input_x_right')
        self.input_y = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='output_y')
        self.dropout_keep_prob = tf.placeholder(shape=[], dtype=tf.float32, name='keep_prob')

        self.length_left = self.get_length(self.input_left)
        self.length_right = self.get_length(self.input_right)


        with tf.name_scope('embedding'):
            # 定义成类的属性，方便训练时直接加载预训练词向量
            self.W = tf.get_variable('embedding_weight', dtype=tf.float32, shape=[vocab_size, embedding_size],
                                     initializer=tf.truncated_normal_initializer())

            self.embedding_chars_left = tf.nn.embedding_lookup(self.W, self.input_left)
            self.embedding_chars_right = tf.nn.embedding_lookup(self.W, self.input_right)

        with tf.name_scope('sequence_encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden)

            output_left, states_left = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                       cell_bw,
                                                                       self.embedding_chars_left,
                                                                       dtype=tf.float32,
                                                                       sequence_length=self.length_left)

            output_right, states_right = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                         cell_bw,
                                                                         self.embedding_chars_right,
                                                                         dtype=tf.float32,
                                                                         sequence_length=self.length_right)

            self.H_left = tf.concat(output_left, -1)
            self.H_right = tf.concat(output_right, -1)
            
        with tf.name_scope('self_attention'):
            W_s1 = tf.get_variable('W_s1', shape=[2*num_hidden, num_da], dtype=tf.float32,
                                   initializer=initializer, regularizer=regularizer)
            W_s2 = tf.get_variable('W_s2', shape=[num_da, num_r], dtype=tf.float32,
                                   initializer=initializer, regularizer=regularizer)

            H_left_reshape = tf.reshape(self.H_left, [-1, 2*num_hidden])
            H_right_reshape = tf.reshape(self.H_right, [-1, 2*num_hidden])

            tmp_left = tf.nn.tanh(tf.matmul(H_left_reshape, W_s1))
            A_left = tf.transpose(tf.reshape(tf.matmul(tmp_left, W_s2), [-1, max_len_left, num_r]), [0, 2, 1])
            self.A_left = get_mask_attention(A_left, self.length_left, max_len_left)

            tmp_right = tf.nn.tanh(tf.matmul(H_right_reshape, W_s1))
            A_right = tf.transpose(tf.reshape(tf.matmul(tmp_right, W_s2), [-1, max_len_right, num_r]), [0, 2, 1])
            self.A_right = get_mask_attention(A_right, self.length_right, max_len_right)

        with tf.name_scope('sentence_embedding'):
            self.M_left = tf.matmul(self.A_left, self.H_left)
            self.M_right = tf.matmul(self.A_right, self.H_right)

        with tf.name_scope('interaction'):
            self.Matching_matrix = tf.concat([self.M_left, self.M_right], 2)
            matrix_flat = tf.reshape(self.Matching_matrix, [-1, num_r*4*num_hidden])

        with tf.name_scope('mlp'):
            W_mlp = tf.get_variable('weight_mlp', dtype=tf.float32, shape=[num_r*4*num_hidden, num_mlp_hidden],
                                    initializer=initializer, regularizer=regularizer)
            b_mlp = tf.get_variable('bias_mlp', dtype=tf.float32, shape=[num_mlp_hidden],
                                    initializer=tf.zeros_initializer(), regularizer=regularizer)
            mlp_relu = tf.nn.relu(tf.matmul(matrix_flat, W_mlp) + b_mlp)
            self.mlp_drop = tf.nn.dropout(mlp_relu, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W_out =  tf.get_variable('weight_out', dtype=tf.float32, shape=[num_mlp_hidden, 2],
                                    initializer=initializer)
            b_out = tf.get_variable('bias_out', dtype=tf.float32, shape=[2],
                                    initializer=tf.zeros_initializer())
            self.scores = tf.matmul(self.mlp_drop, W_out) + b_out

        with tf.name_scope('penalty'):
            I_left = tf.reshape(tf.tile(tf.eye(num_r), [tf.shape(self.A_left)[0], 1]), [-1, num_r, num_r])

            I_right = tf.reshape(tf.tile(tf.eye(num_r), [tf.shape(self.A_right)[0], 1]), [-1, num_r, num_r])

            self.penalty_left = tf.reduce_mean(tf.square(tf.norm(tf.matmul(self.A_left, tf.transpose(self.A_left, [0, 2, 1])) - I_left,
                                                           ord='fro', axis=[-2, -1])))
            self.penalty_right = tf.reduce_mean(tf.square(tf.norm(tf.matmul(self.A_right, tf.transpose(self.A_right, [0, 2, 1])) - I_right,
                                                            ord='fro', axis=[-2, -1])))

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy) + 0.1*(self.penalty_left + self.penalty_right) \
                        + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope('accuracy'):
            prediction_y = tf.argmax(tf.nn.softmax(self.scores), 1)
            acc = tf.equal(prediction_y, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(acc, tf.float32), name='accuracy')

    @staticmethod
    def get_length(x):
        x_sign = tf.sign(tf.abs(x))
        length = tf.reduce_sum(x_sign, axis=1)
        return tf.cast(length, tf.int32)
