# -*- coding:utf-8 -*-

import codecs
import numpy as np
from matplotlib import pyplot as plt

def get_max_length(input_file):
    left_length = []
    right_length = []
    with codecs.open(input_file, 'r', encoding='utf-8_sig') as rfile:
        for line in rfile.readlines():
            data = line.split('\t')
            left_data = data[0].split()
            left_length.append(len(left_data))
            right_data = data[1].split()
            right_length.append(len(right_data))
    return max(max(left_length), max(right_length))


def get_vocab(input_file):
    vocab = set('pad')
    with codecs.open(input_file, 'r', encoding='utf-8_sig') as rfile:
        for line in rfile.readlines():
            data = line.split('\t')
            for char in data[0].split():
                vocab.add(char)
            for char in data[1].split():
                vocab.add(char)
    vocab = {word:(i+1) for i, word in enumerate(vocab)}
    vocab['pad'] = 0
    return vocab

def padding_sentence(data, max_length, vocab):
    sentence = [vocab[word] for word in data.split()]
    if len(sentence) < max_length:
        sentence = sentence + [vocab['pad']]*(max_length-len(sentence))
    elif len(sentence) > max_length:
        sentence = sentence[:max_length]
    return sentence

def load_data(input_file):
    max_length = get_max_length(input_file)
    vocab = get_vocab(input_file)
    left_data = []
    right_data = []
    label = []
    with codecs.open(input_file, 'r', encoding='utf_8_sig') as rfile:
        for line in rfile.readlines():
            data = line.strip().split('\t')

            left_data.append(padding_sentence(data[0], max_length, vocab))
            right_data.append(padding_sentence(data[1], max_length, vocab))
            if int(data[2]) == 0: label.append([1, 0])
            else: label.append([0, 1])
    x_left_data = np.array(left_data)
    x_right_data = np.array(right_data)
    y_label = np.array(label)
    return x_left_data, x_right_data, y_label, vocab, max_length

if __name__ == '__main__':
    # x_left_data, x_right_data, y_label, vocab, max_length = load_data('data/atec_train_data.txt')
    # print(x_left_data[0])
    # print(y_label[0])
    import tensorflow as tf
    dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia')



