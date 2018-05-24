import tensorflow as tf
import numpy as np
import random
import time
import logging,datetime
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import utils
import os,sys
slim=tf.contrib.slim
FLAGS=utils.FLAGS
from densenet import *

num_classes=utils.num_classes
max_timesteps=40
num_features=utils.num_features

def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths):
    """
    multi layer bidirectional rnn
    :param RNN: RNN class, e.g. LSTMCell
    :param num_units: int, hidden unit of RNN cell
    :param num_layers: int, the number of layers
    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size
    :param batch_size: int
    :return: the output of last layer bidirectional rnn with concatenating
    """
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              dtype=tf.float32)
            _inputs = tf.concat(output, 2)

    return _inputs

class Graph(object):
    def __init__(self,is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, utils.image_width, utils.image_height, 1])
            if FLAGS.Use_CRNN:
                with tf.variable_scope('CNN'):
                    self.keep_prob_cv1 = tf.placeholder("float")
                    self.keep_prob_cv2 = tf.placeholder("float")
                    self.keep_prob_cv3 = tf.placeholder("float")
                    self.keep_prob_cv4 = tf.placeholder("float")
                    net = slim.conv2d(self.inputs, 64, [3, 3], scope='conv1')
                    net = tf.nn.dropout(net, self.keep_prob_cv1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='conv3')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    net = tf.nn.dropout(net, self.keep_prob_cv2)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
                    net = tf.nn.dropout(net, self.keep_prob_cv3)
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                    net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
                    net = slim.conv2d(net, 512, [2, 2], padding='VALID', activation_fn=None, scope='conv7')
                    net = tf.nn.dropout(net, self.keep_prob_cv4)
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    self.cnn_time = net.get_shape().as_list()[1]
                    self.num_feauture=512
            else:
                with tf.variable_scope('Dense_CNN'):
                    nb_filter = 64
                    net = tf.layers.conv2d(self.inputs, nb_filter, 5, (2, 2), "SAME", use_bias=False)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    net, nb_filter = transition_block(net, 128, is_training, pooltype=2)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    net, nb_filter = transition_block(net, 128, is_training, pooltype=3)
                    net, nb_filter = dense_block(net, 8, 8, nb_filter, is_training)
                    #net, nb_filter = transition_block(net, 128, is_training, pooltype=3)
                    print(net)
                    #net = tf.layers.conv2d(net, nb_filter, 3, (1, 2), "SAME", use_bias=True)
                    self.cnn_time = net.get_shape().as_list()[1]
                    self.num_feauture=4*192



            temp_inputs = net
            with tf.variable_scope('BLSTM'):
                self.labels = tf.sparse_placeholder(tf.int32)
                self.seq_len=tf.placeholder(tf.int32,[None])
                self.lstm_inputs = tf.reshape(temp_inputs, [-1, self.cnn_time, self.num_feauture])                
                outputs = stacked_bidirectional_rnn(tf.contrib.rnn.LSTMCell, FLAGS.num_hidden, 2,self.lstm_inputs,self.seq_len)
            shape = tf.shape(self.lstm_inputs)
            batch_s, max_timesteps = shape[0], 40
            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden*2])
            self.keep_prob_fc = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(outputs,self.keep_prob_fc)
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden*2,num_classes],stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))
            logits = tf.matmul(outputs, W) + b
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            self.logits_before_ctc = tf.argmax(logits,2)
            logits = tf.transpose(logits, (1, 0, 2))
            self.global_step = tf.Variable(0, trainable=False)
            print("###########################################################")
            print(self.labels)
            print(logits)
            print(self.seq_len)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,self.global_step,FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum, use_nesterov=True).minimize(self.cost, global_step=self.global_step)
            #self.optimizer = tf.train.AdamOptimizer(1e-2).minimize(self.cost, global_step=self.global_step) 
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summay = tf.summary.merge_all()
