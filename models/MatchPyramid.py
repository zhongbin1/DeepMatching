# -*- coding:utf-8 -*-

"""

A tensorflow implementation for text matching
model in paper MatchPyramid.
author: Bin Zhong
data: 2018-11-13

"""

import tensorflow as tf

class MatchPyramid(object):
    def __init__(
      self, max_len_left, max_len_right, vocab_size,
      embedding_size, filter_size, num_filters, num_hidden, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, max_len_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, max_len_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer for both CNN
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars_left = tf.nn.embedding_lookup(W, self.input_left)
            self.embedded_chars_right = tf.nn.embedding_lookup(W, self.input_right)

        with tf.name_scope('interaction'):
            matching_matrix = tf.matmul(self.embedded_chars_left, tf.transpose(self.embedded_chars_right, [0,2,1]))
            self.interaction_exp = tf.expand_dims(matching_matrix, -1)

        # Create a convolution + maxpool layer for each filter size
        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):
        filter_shape = [filter_size, filter_size, 1, num_filters]
        with tf.name_scope("conv-maxpool-layer1"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.interaction_exp,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled_layer1 = tf.nn.max_pool(
                h,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name="pool")
            # pooled_outputs.append(pooled)

        filter_shape = [filter_size-1, filter_size-1, num_filters, num_filters]
        with tf.name_scope("conv-maxpool-layer2"):
            # Convolution Layer
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                pooled_layer1,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled_layer2 = tf.nn.max_pool(
                h,
                ksize=[1, 3, 3, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            self.pooled_reshape = tf.reshape(pooled_layer2, [-1, num_filters])
           

        # hidden layer
        with tf.name_scope("hidden"):
            W = tf.get_variable(
                "W_hidden",
                shape=[num_filters, num_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")

            self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.pooled_reshape, W, b, name="hidden_output"))

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            #print self.h_drop

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[num_hidden, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

