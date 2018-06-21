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
max_long_word = utils.max_long_word

def stacked_bidirectional_rnn(RNN, num_units, num_layers, inputs, seq_lengths):
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            print("_inputs",_inputs.get_shape().as_list())
            rnn_cell_fw = RNN(num_units)
            rnn_cell_bw = RNN(num_units)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    
    with tf.variable_scope('FC'):
        outputs_rnn = _inputs
        num_units_input_from_lstm = outputs_rnn.get_shape().as_list()[2]
        W3 = tf.Variable(tf.truncated_normal([num_units_input_from_lstm,FLAGS.num_units_fc3],stddev=0.1, dtype=tf.float32), name='W3')
        W4 = tf.Variable(tf.truncated_normal([FLAGS.num_units_fc3,num_classes],stddev=0.1, dtype=tf.float32), name='W4')
        b3 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.num_units_fc3], name='b3'))
        b4 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b4'))
        input_from_lstm = tf.reshape(outputs_rnn,[-1,num_units_input_from_lstm])
        neuron = tf.matmul(input_from_lstm, W3) + b3
        neuron = tf.nn.relu(neuron)
        neuron = tf.matmul(neuron, W4) + b4
        print(neuron.get_shape().as_list())
        outputs_final_fc = tf.reshape(neuron,[FLAGS.batch_size, -1,num_classes])
        #print(outputs_final_fc.get_shape().as_list())
    return outputs_final_fc

def base_rnn_2u(RNN, num_units, num_layers, inputs, seq_lengths):
    _inputs = inputs
    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for _ in range(num_layers):
        print("lstm shpe",_inputs.get_shape().as_list())
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = RNN(num_units)
            (output, state) = tf.nn.dynamic_rnn(rnn_cell_fw, _inputs, dtype=tf.float32)
            _inputs = output
    with tf.variable_scope('FC'):
        outputs_rnn = _inputs
        num_units_input_from_lstm = outputs_rnn.get_shape().as_list()[2]
        W3 = tf.Variable(tf.truncated_normal([num_units_input_from_lstm,num_classes],stddev=0.1, dtype=tf.float32), name='W3')
        b3 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b3'))
        input_from_lstm = tf.reshape(outputs_rnn,[-1,num_units_input_from_lstm])
        neuron = tf.matmul(input_from_lstm, W3) + b3
        outputs_final_fc = tf.reshape(neuron,[FLAGS.batch_size, -1,num_classes])
        
    return outputs_final_fc

def compute_cost(real_word_index, predict_word_index):
    cost = tf.Variable(initial_value = 0.0, dtype=tf.float32)
    real_word_one_hot = tf.one_hot(real_word_index,num_classes)
    predict_word_softmax = tf.nn.softmax(predict_word_index)
    cost = - tf.reduce_sum(real_word_one_hot * tf.log(predict_word_softmax))
    return cost

class Graph(object):
    def __init__(self,is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [FLAGS.batch_size, utils.image_width, utils.image_height, 1])
            self.keep_prob = tf.placeholder("float")
            if FLAGS.Use_CRNN:
                with tf.variable_scope('CNN'):
                    self.keep_prob_cv = tf.placeholder("float")
                    net = slim.conv2d(self.inputs, 64, [3, 3], scope='conv1')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = tf.nn.dropout(net, self.keep_prob)
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv6')
                    net = tf.nn.dropout(net, self.keep_prob)
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.conv2d(net, 512, [3, 3], scope='conv7')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv8')
                    #net = slim.conv2d(net,128,[1,1],scope='conv9')
                    net = tf.nn.dropout(net, self.keep_prob)
                    self.cnn_time = net.get_shape().as_list()[1]
                    self.num_feature_of_fc = net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]
                    fc_input = tf.reshape(net,[-1, self.num_feature_of_fc])
                    print("self.num_feature_of_fc ", self.num_feature_of_fc)
                #RRAM_recude
                with tf.variable_scope('FC'):
                    W1 = tf.Variable(tf.truncated_normal([self.num_feature_of_fc,FLAGS.num_units_fc1],stddev=0.1, dtype=tf.float32), name='W1')
                    W2 = tf.Variable(tf.truncated_normal([FLAGS.num_units_fc1,FLAGS.num_units_fc2],stddev=0.1, dtype=tf.float32), name='W2')
                    b1 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.num_units_fc1], name='b1'))
                    b2 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.num_units_fc2], name='b2'))
                    neuron = tf.matmul(fc_input, W1) + b1
                    neuron = tf.layers.batch_normalization(neuron, training=is_training)
                    neuron = tf.nn.relu(neuron)
                    neuron = tf.nn.dropout(neuron, self.keep_prob)
                    neuron = tf.matmul(neuron, W2) + b2
                    neuron = tf.layers.batch_normalization(neuron, training=is_training)
                    rnn_input = tf.nn.relu(neuron)
                    rnn_input = tf.nn.dropout(rnn_input, self.keep_prob)
                    print("output_fc",rnn_input.get_shape().as_list())
                #RRAM
                '''
                with tf.variable_scope('FC'):
                    W1 = tf.Variable(tf.truncated_normal([self.num_feature_of_fc,FLAGS.num_units_fc1],stddev=0.1, dtype=tf.float32), name='W1')
                    W2 = tf.Variable(tf.truncated_normal([FLAGS.num_units_fc1,FLAGS.num_units_fc2],stddev=0.1, dtype=tf.float32), name='W2')
                    b1 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.num_units_fc1], name='b1'))
                    b2 = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.num_units_fc2], name='b2'))
                    neuron = tf.matmul(fc_input, W1) + b1
                    neuron = tf.nn.relu(neuron)
                    neuron = tf.matmul(neuron, W2) + b2
                    rnn_input = tf.nn.relu(neuron)
                    print(rnn_input.get_shape().as_list())'''

            with tf.variable_scope('BLSTM'):
                self.labels = tf.placeholder(tf.int32,[None,max_long_word])
                #self.seq_len=tf.placeholder(tf.int32,[None])
                #self.lstm_inputs = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[FLAGS.batch_size,FLAGS.time_step,FLAGS.num_units_fc2], name='lstm_inputs'))
                input_lstm = []
                for j in range(FLAGS.batch_size):
                    for i in range(FLAGS.time_step):
                        input_lstm.append(rnn_input[j])
                self.lstm_inputs = tf.reshape(tf.stack(input_lstm), [-1,FLAGS.time_step,FLAGS.num_units_fc2])              
                outputs_rnn = base_rnn_2u(tf.contrib.rnn.LSTMCell, FLAGS.num_hidden, FLAGS.num_layers_rnn,self.lstm_inputs,FLAGS.time_step)
                print("output_rnn:",outputs_rnn.get_shape().as_list())

            with tf.variable_scope('cost'):  
                self.cost_batch = compute_cost(self.labels, outputs_rnn)
                self.cost_batch_avg = tf.div(self.cost_batch,FLAGS.batch_size)
                self.global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,self.global_step,FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True)
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum, use_nesterov=True).minimize(self.cost_batch_avg,
                                                                                                    global_step=self.global_step)

            with tf.variable_scope('acc'):
                self.logits = tf.argmax(outputs_rnn,2)


           

'''                                                                                                            global_step=self.global_step)
           
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summay = tf.summary.merge_all()
'''
