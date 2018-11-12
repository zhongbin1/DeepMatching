# -*- coding:utf-8 -*-

import os
import time
import datetime
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
from models.L2R import L2R
from models.MVLSTM import MVLSTM


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
tf.flags.DEFINE_integer("num_k", 10, "Number of k (default: 5)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
# Data Parameter
tf.flags.DEFINE_integer("max_len_left", 10, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 10, "max document length of right input")
tf.flags.DEFINE_integer("most_words", 300000, "Most number of words in vocab (default: 300000)")
# Training parameters
tf.flags.DEFINE_integer("seed", 123, "Random seed (default: 123)")
tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_float("eval_split", 0.1, "Use how much data for evaluating (default: 0.1)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(FLAGS.most_words-1)]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.append('<UNK/>')
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(data_left, data_right, label, vocab):
    vocabset = set(vocab.keys())
    out_left = np.array([[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence ] for sentence in data_left])
    out_right = np.array([[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence ] for sentence in data_right])
    out_y = np.array([[0, 1] if x == 1 else [1, 0] for x in label])
    return [out_left, out_right, out_y]

def load_data(filepath, vocab_tuple=None):
    data = list(open(filepath, "r", encoding='utf-8').readlines())
    data = [s.strip().split(',') for s in data]
    
    # Split by words
    data = list(filter(lambda x: len(x)==3, data))
    data_left = [x[0].strip().split(' ') for x in data]
    data_right = [x[1].strip().split(' ') for x in data]
    data_label = [x[2] for x in data]
    data_label = list(map(int, data_label))
    num_pos = sum(data_label)
    data_left = pad_sentences(data_left, FLAGS.max_len_left)
    data_right = pad_sentences(data_right, FLAGS.max_len_right)
    if vocab_tuple is None:
        vocab, vocab_inv = build_vocab(data_left+data_right)
    else:
        vocab, vocab_inv = vocab_tuple
    data_left, data_right, data_label = build_input_data(data_left, data_right, data_label, vocab)
    return data_left, data_right, data_label, vocab, vocab_inv, num_pos

def main():
    # Load data
    print("Loading data...")
    x_left_train, x_right_train, y_train, vocab, vocab_inv, num_pos = load_data(os.path.join(FLAGS.train_dir, 'data/train.txt'))
    x_left_dev, x_right_dev, y_dev, vocab, vocab_inv, num_pos = load_data(os.path.join(FLAGS.train_dir, 'data/test.txt'), (vocab, vocab_inv))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = L2R(
                max_len_left=FLAGS.max_len_left,
                max_len_right=FLAGS.max_len_right,
                vocab_size=len(vocab),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                num_hidden=FLAGS.num_hidden,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            #model = MVLSTM(
                # max_len_left=FLAGS.max_len_left,
                # max_len_right=FLAGS.max_len_right,
                # vocab_size=len(vocab),
                # embedding_size=FLAGS.embedding_dim,
                # num_k=FLAGS.num_k,
                # num_hidden=FLAGS.num_hidden,
                # l2_reg_lambda=FLAGS.l2_reg_lambda)


            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Write the computation graph
            writer = tf.summary.FileWriter('../logs/', sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs", timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            def batch_iter(all_data, batch_size, num_epochs, shuffle=True):
                data = np.array(all_data)
                data_size = len(data)
                num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
                for epoch in range(num_epochs):
                    # Shuffle the data at each epoch
                    if shuffle:
                        shuffle_indices = np.random.permutation(np.arange(data_size))
                        shuffled_data = data[shuffle_indices]
                    else:
                        shuffled_data = data
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batch_num * batch_size
                        end_index = min((batch_num + 1) * batch_size, data_size)
                        yield shuffled_data[start_index:end_index]


            def train_step(x_left_batch, x_right_batch, y_batch):
                feed_dict = {
                model.input_left: x_left_batch,
                model.input_right: x_right_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            # def dev_step(x_left_batch_dev, x_right_batch_dev, y_batch_dev):
            #     feed_dict = {
            #     cnn.input_left: x_left_batch_dev,
            #     cnn.input_right: x_right_batch_dev,
            #     cnn.input_y: y_batch_dev,
            #     cnn.dropout_keep_prob: 1.0
            #     }
            #     step, loss, accuracy, sims, pres = sess.run(
            #             [global_step, cnn.loss, cnn.accuracy, cnn.sims, cnn.scores],
            #             feed_dict)
            #     return loss,accuracy

            def dev_whole(x_left_dev, x_right_dev, y_dev):
                feed_dict = {
                model.input_left: x_left_dev,
                model.input_right: x_right_dev,
                model.input_y: y_dev,
                model.dropout_keep_prob: 1.0
                }
                step, loss, accuracy, pres = sess.run(
                    [global_step, model.loss, model.accuracy, model.scores],
                    feed_dict)
                return loss, accuracy

            # The model is considered overfit if accuracy decrease for four times continously
            def overfit(dev_acc):
                n = len(dev_acc)
                if n < 5:
                    return False
                for i in range(n-4, n):
                    if dev_acc[i] > dev_acc[i-1]:
                        return False
                return True

            # Generate batches
            batches = batch_iter(
                list(zip(x_left_train, x_right_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            dev_acc = []
            for batch in batches:
                x1_batch, x2_batch, y_batch = zip(*batch)
                train_step(x1_batch, x2_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    loss, accuracy = dev_whole(x_left_dev, x_right_dev, y_dev)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: dev-result, loss {:g}, acc {:g}".format(time_str, loss, accuracy))
                    dev_acc.append(accuracy)
                    print("Recently accuracy:")
                    print(dev_acc[-10:])
                    if overfit(dev_acc):
                        print('Overfit!!')
                        break
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    main()
