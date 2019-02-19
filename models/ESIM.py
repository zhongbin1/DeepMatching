# -*- coding:utf-8 -*-

import tensorflow as tf

class ESIM(object):
    def __init__(self, max_len_left, max_len_right, vocab_size,
                embedding_size, rnn_size, num_hidden, l2_reg_lambda=0.0):

        # placeholder for input data
        self.input_left  = tf.placeholder(tf.int32, shape=[None, max_len_left],
                                         name="input_left")
        self.input_right = tf.placeholder(tf.int32, shape=[None, max_len_right],
                                         name="input_output")
        self.input_y     = tf.placeholder(tf.float32, shape=[None, 2],
                                         name="input_y")
        self.keep_prob   = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("input_embedding"):
            self.embedding_weight = tf.get_variable("embedding_weight",
                                                    shape=[vocab_size, embedding_size],
                                                    dtype=tf.float32,
                                                    initializer=tf.truncated_normal_initializer())
            self.emb_left  = tf.nn.embedding_lookup(self.embedding_weight, self.input_left, name="emb_left")
            self.emb_right = tf.nn.embedding_lookup(self.embedding_weight, self.input_right, name="emb_right")

        with tf.name_scope("bilstm"):
            self.length_left  = self.get_length(self.input_left)
            self.length_right = self.get_length(self.input_right)

            cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="bilstm")
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="bilstm")
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

            (output_fw_left, output_bw_left), states_left   = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                            cell_bw,
                                                                                            self.emb_left,
                                                                                            dtype=tf.float32,
                                                                                            sequence_length=self.length_left)

            (output_fw_right, output_bw_right), states_right = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                               cell_bw,
                                                                                               self.emb_right,
                                                                                               dtype=tf.float32,
                                                                                               sequence_length=self.length_right)

            self.H_left  = tf.concat([output_fw_left, output_bw_left], 2)
            self.H_right = tf.concat([output_fw_right, output_bw_right], 2)


        with tf.name_scope("local_inference"):
            self.E = tf.matmul(self.H_left, tf.transpose(self.H_right, [0, 2, 1]), name="soft_alignment")
            self.att_left  = tf.nn.softmax(self.E, 2, name="left_attention_weight")
            self.att_right = tf.nn.softmax(self.E, 1, name="left_attention_weight")

            self.tilde_a = tf.matmul(self.att_left, self.H_right, name="tilde_a")
            self.tilde_b = tf.matmul(tf.transpose(self.att_right, [0, 2, 1]), self.H_left, name="tilde_b")

            local_inf_left  = [self.H_left, self.tilde_a, self.H_left - self.tilde_a, tf.multiply(self.H_left, self.tilde_a)]
            local_inf_right = [self.H_right, self.tilde_b, self.H_right - self.tilde_b, tf.multiply(self.H_right, self.tilde_b)]

            self.M_left  = tf.concat(local_inf_left, 2, name="m_left")
            self.M_right = tf.concat(local_inf_right, 2, name="m_right")

        with tf.name_scope("feedforward_layer"):
            W_feed = tf.get_variable("W_feed", shape=[rnn_size*8, rnn_size], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())
            b_feed = tf.get_variable("b_feed", shape=[rnn_size], dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            self.F_left  = tf.nn.relu(tf.einsum('aij,jk->aik', self.M_left, W_feed) + b_feed, name="F_left")
            self.F_right = tf.nn.relu(tf.einsum('aij,jk->aik', self.M_right, W_feed) + b_feed, name="F_right")

        with tf.name_scope("composition_layer"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="bilstm_cross")
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True, name="bilstm_cross")
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

            (output_fw_left, output_bw_left), states_left    = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                                cell_bw,
                                                                                                self.F_left,
                                                                                                dtype=tf.float32,
                                                                                                sequence_length=self.length_left)

            (output_fw_right, output_bw_right), states_right = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                               cell_bw,
                                                                                               self.F_right,
                                                                                               dtype=tf.float32,
                                                                                               sequence_length=self.length_right)

            self.V_left  = tf.concat([output_fw_left, output_bw_left], 2)
            self.V_right = tf.concat([output_fw_right, output_bw_right], 2)

        with tf.name_scope("pooling"):
            self.V_left_ave = tf.reduce_mean(self.V_left, 1, keep_dims=False)
            self.V_left_max = tf.reduce_max(self.V_left, 1, keep_dims=False)

            self.V_right_ave = tf.reduce_mean(self.V_right, 1, keep_dims=False)
            self.V_right_max = tf.reduce_max(self.V_right, 1, keep_dims=False)

            self.V_concat = tf.concat([self.V_left_ave, self.V_left_max, self.V_right_ave, self.V_right_max],
                                      axis=1)

        l2_loss = 0.0
        with tf.name_scope("mlp_layer"):
            W = tf.get_variable("w_mlp", shape=[rnn_size*8, num_hidden], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable("b_mlp", shape=[num_hidden], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            full_out = tf.nn.xw_plus_b(self.V_concat, W, b)
            full_out = tf.nn.tanh(full_out)
            self.full_out = tf.nn.dropout(full_out, keep_prob=self.keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b_output")
            self.scores = tf.nn.xw_plus_b(self.full_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")



    @staticmethod
    def get_length(x):
        x_sign = tf.sign(tf.abs(x))
        length = tf.reduce_sum(x_sign, axis=1)
        return tf.cast(length, tf.int32)

if __name__ == "__main__":
    import numpy as np
    a = np.array([[[1,2,0],[2,4,5],[7,8,9]]], dtype=np.float32)
    cell_fw = tf.nn.rnn_cell.LSTMCell(5)
    cell_bw = tf.nn.rnn_cell.LSTMCell(5)
    (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,
                                                                    a,
                                                                    dtype=tf.float32,
                                                                    sequence_length=[2])
    #c = tf.cast()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_fw_, output_bw_, state_ = sess.run([output_fw, output_bw, state])
        print("output_fw", output_fw_.shape)
        print(output_fw_)
        print("output_bw", output_bw_.shape)
        print(output_bw_)
        #print("state", state_.shape())
        print(state_)
