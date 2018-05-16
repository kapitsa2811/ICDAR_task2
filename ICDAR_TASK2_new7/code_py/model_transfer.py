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
from densenet import *
FLAGS=utils.FLAGS
from tensorflow.python.platform import gfile

num_classes=utils.num_classes
max_timesteps=0
num_features=utils.num_features

class Graph(object):
    def __init__(self,is_training=True,pb_file_path=None):
        with gfile.FastGFile(os.path.join(pb_file_path), 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.original_pic, self.seq_len, self.shape_lstm_inputs, self.outputs_from_rnn = tf.import_graph_def(self.graph_def, return_elements=['original_pic:0', 'seq_len:0', 'shape_lstm_inputs:0', 'output_rnn:0'])       
            self.labels = tf.sparse_placeholder(tf.int32)     
            shape = self.shape_lstm_inputs
            batch_s, max_timesteps = shape[0], shape[1]
            W = tf.Variable(tf.truncated_normal([FLAGS.num_hidden*2,num_classes],stddev=0.1, dtype=tf.float32), name='W')
            b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[num_classes], name='b'))
            
            logits = tf.matmul(self.outputs_from_rnn, W) + b
            logits = tf.reshape(logits, [batch_s, -1, num_classes])
            logits = tf.transpose(logits, (1, 0, 2))
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len)
            self.cost = tf.reduce_mean(self.loss)
            self.learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,self.global_step,FLAGS.decay_steps,
                                                            FLAGS.decay_rate, staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=FLAGS.momentum, use_nesterov=True).minimize(self.cost,global_step=self.global_step)
           
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len, merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
            self.lerr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

            tf.summary.scalar('cost', self.cost)
            self.merged_summay = tf.summary.merge_all()
            
