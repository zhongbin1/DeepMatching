# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import layers


def dense_layer(inputs, kernel_size, num_filters, keep_prob, scope, regularizer=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_size = inputs.get_shape()[-1]  # [batch_size, max_len, input_size]
        W = tf.get_variable("W", shape=[kernel_size, input_size, num_filters], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        b = tf.get_variable("b", shape=[num_filters], dtype=tf.float32,
                            initializer=tf.zeros_initializer(), regularizer=regularizer)
        outputs = tf.nn.conv1d(inputs, W, 1, padding='SAME') + b
        outputs = tf.nn.leaky_relu(outputs)
        outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)  # [batch_size, max_len, num_filters]
    return outputs

def fc_layer(inputs, output_dim, keep_prob, scope, regularizer=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_size = inputs.get_shape()[-1]
        W = tf.get_variable("W", shape=[input_size, output_dim], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
        b = tf.get_variable("b", shape=[output_dim], dtype=tf.float32,
                            initializer=tf.zeros_initializer(), regularizer=regularizer)

        if inputs.shape.ndims == 3:
            outputs = tf.einsum("abc,cd->abd", inputs, W) + b
        else:
            outputs = tf.matmul(inputs, W) + b
        outputs = tf.nn.leaky_relu(outputs)
        outputs = tf.nn.dropout(outputs, keep_prob)
    return outputs, W

def get_masked_weights(inputs, seq_len, max_len):
    seq_mask = tf.sequence_mask(seq_len, max_len, dtype=tf.float32)  # [batch_size, max_len]
    seq_mask = tf.expand_dims(seq_mask, 1)  # [batch_size, 1, max_len]
    outputs = inputs * seq_mask + (seq_mask - 1) * 1e9
    outputs = tf.nn.softmax(outputs, axis=-1)
    return outputs


class DSA(object):
    def __init__(self, max_len_left, max_len_right, vocab_size,
                 embedding_size, num_hidden,
                 d_1, d_l, k_1, k_2, num_layers, d_c,
                 num_attentions, d_o, num_iter, mu=1e-2, l2_reg_lambda=0.0):

        regularizer = layers.l2_regularizer(l2_reg_lambda)

        # placeholder for input data
        self.input_left = tf.placeholder(tf.int32, shape=[None, max_len_left],
                                         name="input_left")
        self.input_right = tf.placeholder(tf.int32, shape=[None, max_len_right],
                                          name="input_right")
        self.input_y = tf.placeholder(tf.float32, shape=[None, 2],
                                      name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            self.embedding_weight = tf.get_variable("embedding_weight",
                                                    shape=[vocab_size, embedding_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer())
            self.emb_left = tf.nn.embedding_lookup(self.embedding_weight, self.input_left, name="emb_left")
            self.emb_right = tf.nn.embedding_lookup(self.embedding_weight, self.input_right, name="emb_right")

            self.length_left = self.get_length(self.input_left)
            self.length_right = self.get_length(self.input_right)

        with tf.name_scope('dense_layers'):
            X_1_left  = dense_layer(self.emb_left, 1, d_1, self.dropout_keep_prob, 'dense_layer_1', regularizer)
            X_1_right = dense_layer(self.emb_right, 1, d_1, self.dropout_keep_prob, 'dense_layer_1', regularizer)

            k_size_list = [k_1, k_2]
            layer_outputs_left = [[X_1_left], [X_1_left]]
            layer_outputs_right = [[X_1_right], [X_1_right]]
            for k in range(2):
                for l in range(2, num_layers + 1):
                    temp_inputs_left  = tf.concat(layer_outputs_left[k], axis=-1)
                    temp_inputs_right = tf.concat(layer_outputs_right[k], axis=-1)

                    X_i_left  = dense_layer(temp_inputs_left, k_size_list[k], d_l, self.dropout_keep_prob,
                                           'dense_layer_{}_{}'.format(k, l), regularizer)
                    X_i_right = dense_layer(temp_inputs_right, k_size_list[k], d_l, self.dropout_keep_prob,
                                           'dense_layer_{}_{}'.format(k, l), regularizer)

                    layer_outputs_left[k].append(X_i_left)
                    layer_outputs_right[k].append(X_i_right)

            concat_outputs_left  = [self.emb_left] + layer_outputs_left[0] + layer_outputs_left[1]
            concat_outputs_right = [self.emb_right] + layer_outputs_right[0] + layer_outputs_right[1]

            self.X_c_left  = dense_layer(tf.concat(concat_outputs_left, -1), 1, d_c, self.dropout_keep_prob,
                                   'dense_layer_c', regularizer)
            self.X_c_right = dense_layer(tf.concat(concat_outputs_right, -1), 1, d_c, self.dropout_keep_prob,
                                   'dense_layer_c', regularizer)


        with tf.name_scope('dynamic_self_attention'):
            Z_left = []
            Z_right = []
            W_j = []
            for j in range(num_attentions):
                X_hat_left, W  = fc_layer(self.X_c_left, d_o, 1.0, 'dsa_{}'.format(j), regularizer)
                X_hat_right, _ = fc_layer(self.X_c_right, d_o, 1.0, 'dsa_{}'.format(j), regularizer)

                q_left = tf.zeros(shape=[tf.shape(X_hat_left)[0], 1, max_len_left], dtype=tf.float32)
                q_right = tf.zeros(shape=[tf.shape(X_hat_right)[0], 1, max_len_right], dtype=tf.float32)

                for r in range(num_iter):
                    a_left = get_masked_weights(q_left, self.length_left, max_len_left)
                    s_left = tf.matmul(a_left, X_hat_left)  # [batch_size, 1, d_o]
                    z_left = tf.nn.tanh(s_left)

                    a_right = get_masked_weights(q_right, self.length_right, max_len_right)
                    s_right = tf.matmul(a_right, X_hat_right)  # [batch_size, 1, d_o]
                    z_right = tf.nn.tanh(s_right)

                    if r == num_iter-1:
                        Z_left.append(tf.reshape(z_left, shape=[-1, d_o]))
                        Z_right.append(tf.reshape(z_right, shape=[-1, d_o]))
                        # for visualize
                        att_left = tf.identity(a_left, name='attention_left')
                        att_right = tf.identity(a_right, name='attention_right')

                    X_left_temp = X_hat_left / tf.sqrt(tf.reduce_sum(tf.square(X_hat_left), axis=-1, keepdims=True))
                    z_left_temp = z_left / tf.sqrt(tf.reduce_sum(tf.square(z_left), axis=-1, keepdims=True))
                    X_right_temp = X_hat_right / tf.sqrt(tf.reduce_sum(tf.square(X_hat_right), axis=-1, keepdims=True))
                    z_right_temp = z_right / tf.sqrt(tf.reduce_sum(tf.square(z_right), axis=-1, keepdims=True))

                    q_left = q_left + tf.matmul(z_left_temp, tf.transpose(X_left_temp, [0, 2, 1]))
                    q_right = q_right + tf.matmul(z_right_temp, tf.transpose(X_right_temp, [0, 2, 1]))

                W_j.append(W)


        with tf.name_scope('penalization'):
            self.penalty = 0.0
            for i in range(num_attentions):
                for j in range(i+1, num_attentions):
                    self.penalty += tf.nn.relu(1 - tf.square(tf.norm(W_j[i]-W_j[j], ord='fro', axis=[0, 1])))


        with tf.name_scope('mlp_layer'):
            self.V_left  = tf.concat(Z_left, axis=-1)
            self.V_right = tf.concat(Z_right, axis=-1)
            self.V = tf.concat([self.V_left, self.V_right, tf.abs(self.V_left-self.V_right),
                                tf.multiply(self.V_left, self.V_right)], axis=-1)

            output, _ = fc_layer(self.V, num_hidden, self.dropout_keep_prob, 'fc_1', regularizer=regularizer)
            # has a shortcut connection
            self.full_out, _ = fc_layer(tf.concat([self.V, output], axis=-1), num_hidden, self.dropout_keep_prob, 'fc_2',
                                     regularizer=regularizer)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2], dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
            b = tf.get_variable("b_output", shape=[2], dtype=tf.float32, initializer=tf.constant_initializer(0.1),
                                regularizer= regularizer)
            self.scores = tf.nn.xw_plus_b(self.full_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + mu * self.penalty + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")


    @staticmethod
    def get_length(x):
        x_sign = tf.sign(tf.abs(x))
        length = tf.reduce_sum(x_sign, axis=1)
        return tf.cast(length, tf.int32)

if __name__ == '__main__':
    a = tf.get_variable('a', shape=[2, 3, 4], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('b', shape=[2, 1, 4], dtype=tf.float32, initializer=tf.truncated_normal_initializer())

    a = a/tf.sqrt(tf.reduce_sum(tf.square(a), axis=-1, keepdims=True))
    b = b/tf.sqrt(tf.reduce_sum(tf.square(b), axis=-1, keepdims=True))

    c = tf.matmul(b, tf.transpose(a, [0, 2, 1]))
    d = tf.expand_dims(tf.reduce_sum(tf.multiply(a, b), axis=-1), 1)

    e = tf.constant([1, 1, 2**0.5], dtype=tf.float32)
    f = e/tf.sqrt(tf.reduce_sum(tf.square(e), axis=-1, keepdims=True))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c_, d_, f_ = sess.run([c, d, f])
        print(c_)
        print(d_)
        print(f_)