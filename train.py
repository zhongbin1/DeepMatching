# -*- coding:utf-8 -*-

import os
import time
import datetime
import tensorflow as tf
from data_helps import load_data, batch_iter
from utils import get_model


# Model Hyperparameters
tf.flags.DEFINE_string("model_name", "dsa", "model name (default: 'esim')")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 64)")
tf.flags.DEFINE_string("filter_sizes", "2,3", "Comma-separated filter sizes (default: '2,3')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_integer("num_hidden", 512, "Number of hidden layer units (default: 100)")
tf.flags.DEFINE_integer("num_k", 4, "Number of k (default: 5)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0005, "L2 regularizaion lambda (default: 0.0)")
# Data Parameter
tf.flags.DEFINE_integer("max_len_left", 32, "max document length of left input")
tf.flags.DEFINE_integer("max_len_right", 32, "max document length of right input")

# Training parameters
tf.flags.DEFINE_string("train_dir", "./", "Training dir root")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def train():
    # Load data
    print("Loading data...")
    x_left_train, x_right_train, y_train, vocab = load_data('data/LCQMC/LCQMC_train_seg_with_sw.dat', 32)
    x_left_dev, x_right_dev, y_dev, _ = load_data('data/LCQMC/LCQMC_dev_seg_with_sw.dat', 32)
    x_left_test, x_right_test, y_test, _ = load_data('data/LCQMC/LCQMC_test_seg_with_sw.dat', 32)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = get_model(FLAGS, vocab)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.RMSPropOptimizer(0.0004, 0.9)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Checkpoint directory
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(FLAGS.train_dir, "runs/" + FLAGS.model_name, timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print("Writing to {}\n".format(out_dir))
            checkpoint_prefix = os.path.join(out_dir, "model")

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            sess.run(tf.global_variables_initializer())

            def train_step(x_premise, x_hypothesis, targets):
                """A single training step"""
                feed_dict = {
                    model.input_left: x_premise,
                    model.input_right: x_hypothesis,
                    model.input_y: targets,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                }

                _, step, loss, accuracy, predicted_prob = sess.run(
                    [train_op, global_step, model.loss, model.accuracy, model.predictions],
                    feed_dict)

                if step % 10 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            def check_step(dataset, shuffle=False):
                num_test = 0
                num_correct = 0.0
                batches = batch_iter(dataset, FLAGS.batch_size, 1, shuffle=shuffle)
                for batch in batches:
                    x_premise, x_hypothesis, targets = zip(*batch)
                    feed_dict = {
                        model.input_left: x_premise,
                        model.input_right: x_hypothesis,
                        model.input_y: targets,
                        model.dropout_keep_prob: 1.0
                    }
                    batch_accuracy, predicted_prob = sess.run([model.accuracy, model.predictions], feed_dict)
                    num_test += len(predicted_prob)
                    num_correct += len(predicted_prob) * batch_accuracy
                # calculate Accuracy
                acc = num_correct / num_test
                print('num_test_samples: {}  accuracy: {}'.format(num_test, acc))
                return acc


            # Generate batches
            batches = batch_iter(list(zip(x_left_train, x_right_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            best_acc = 0.0
            for batch in batches:
                x_premise, x_hypothesis, targets = zip(*batch)
                train_step(x_premise, x_hypothesis, targets)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation on dev:")
                    valid_acc = check_step(list(zip(x_left_dev, x_right_dev, y_dev)), shuffle=False)
                    print("\nEvaluation on test:")
                    test_acc = check_step(list(zip(x_left_test, x_right_test, y_test)), shuffle=False)
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


def main():
    train()

if __name__ == '__main__':
    tf.app.run()