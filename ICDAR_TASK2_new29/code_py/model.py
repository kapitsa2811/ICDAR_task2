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


#delete
max_timesteps=40

num_classes=utils.num_classes
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

class Graph(object):
    def __init__(self,is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, utils.image_width, utils.image_height, 1])
            self.y_ = tf.placeholder(tf.float32,[None,FLAGS.num_classes_len])
            self.keep_prob_fc = tf.placeholder("float")
            self.keep_prob_cv1 = tf.placeholder("float")
            self.global_step = tf.Variable(0, trainable=False)

            if FLAGS.Use_CRNN:
                with tf.variable_scope('CNN'):
                    net = slim.conv2d(self.inputs, 64, [3, 3], scope='conv1')
                    net = tf.nn.dropout(net, self.keep_prob_cv1)
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope='conv3')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    net = tf.nn.dropout(net, self.keep_prob_cv1)
                    net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool3')
                    net = slim.conv2d(net, 512, [3, 3], activation_fn=None, scope='conv5')
                    net = tf.nn.dropout(net, self.keep_prob_cv1)
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net, 512, [3, 3], scope='conv6')
                    net = slim.max_pool2d(net, [2, 2], [1, 2], scope='pool4')
                    net = slim.conv2d(net, 512, [2, 2], padding='VALID', activation_fn=None, scope='conv7')
                    net = tf.nn.dropout(net, self.keep_prob_cv1)
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    net = slim.conv2d(net, 64, [1, 1], padding='VALID', activation_fn=None, scope='conv8')
                    net = tf.layers.batch_normalization(net, training=is_training)
                    net = tf.nn.relu(net)
                    self.cnn_time = net.get_shape().as_list()
                    self.num_feauture = net.get_shape().as_list()[1] * net.get_shape().as_list()[2] * net.get_shape().as_list()[3]
            net_flatten = tf.reshape(net, [-1, self.num_feauture])
            W_fc1 = weight_variable([self.num_feauture, 1024])
            b_fc1 = bias_variable([1024])
            g_fc1 = tf.matmul(net_flatten, W_fc1) + b_fc1
            g_fc1 = tf.layers.batch_normalization(g_fc1, training=is_training)
            h_fc1 = tf.nn.relu(g_fc1)

            W_fc2 = weight_variable([1024, 128])
            b_fc2 = bias_variable([128])
            g_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
            g_fc2 = tf.layers.batch_normalization(g_fc2, training=is_training)
            h_fc2 = tf.nn.relu(g_fc2)

            W_fc3 = weight_variable([128, FLAGS.num_classes_len])
            b_fc3 = bias_variable([FLAGS.num_classes_len])
            y_num = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

            self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(y_num))
            self.train_step = tf.train.AdamOptimizer(FLAGS.initial_learning_rate).minimize(self.cross_entropy,global_step=self.global_step)
            self.correct_prediction = tf.equal(tf.argmax(self.y_,1), tf.argmax(y_num,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

