import cv2,time,os,re
import tensorflow as tf
import numpy as np
import utils
import model
from numpy import *

g = model.Graph(is_training=True)

#debug3
'''
if True:
    val_labels = [[1,2,3,4,0],[1,2,3,4,0]]
    pred_labels = [[1,2,3,4,1],[9,2,3,4,0]]
    acc = utils.compute_acc(np.array(val_labels),np.array(pred_labels))
    print(acc)
'''

#debug2
'''
charset='! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz~¡¢£©°É•€★'
num_classes=len(charset)+1

encode_maps={}
decode_maps={}
for i,char in enumerate(charset,1):
	print(i,char)
#g = model.Graph(is_training=True)
labels = tf.placeholder(tf.int32,[None,None])
labels_one_hot = tf.one_hot(labels,5)
with tf.Session() as sess:
    val_labels = [[0,2,3,4,5],[1,2,4,4,5]]
    label_one_hot = sess.run(labels_one_hot,{labels:val_labels})
    print(label_one_hot)
'''

'''
#debug1
num_classes=utils.num_classes
FLAGS=utils.FLAGS
train_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train'
train_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt'
val_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train'
val_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt'

labels = tf.placeholder(tf.int32,[None,None])
labels_one_hot = tf.one_hot(labels,num_classes)
print(labels_one_hot)
sss = labels

with tf.Session() as sess:
    val_feeder=utils.DataIterator2(data_dir=val_dir, text_dir=val_text_dir)
    val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
    print('***************get image: ',val_feeder.size)
    print(val_labels)
    label_one_hot = sess.run(labels_one_hot,{labels:val_labels})
    print(type(label_one_hot))
    print(label_one_hot[2][10])
    print(label_one_hot[2][0])
    print(label_one_hot.shape)
'''
