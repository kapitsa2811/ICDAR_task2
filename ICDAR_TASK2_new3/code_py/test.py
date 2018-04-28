import cv2,time,os,re
import tensorflow as tf
import numpy as np
import utils
import model
FLAGS = utils.FLAGS

data_dir = '../test_data/train_words'
text_dir = '../test_data/train_words_gt.txt'
#data_dir = '../test_data/val_words'
#text_dir = '../test_data/val_words_gt.txt'

def acc():
    acc_all = []
    g = model.Graph()
    with tf.Session(graph = g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore') 
        test_feeder=utils.DataIterator2(data_dir=data_dir,text_dir=text_dir)
        print("total data:",test_feeder.size)
        print("total image in folder", test_feeder.total_pic_read)
        total_epoch = int(test_feeder.size / FLAGS.batch_size) + 1
        for cur_batch in range(total_epoch):
            print("cur_epoch/total_epoch",cur_batch,"/",total_epoch)
            indexs=[]
            cur_batch_num = FLAGS.batch_size
            if cur_batch == int(test_feeder.size / FLAGS.batch_size):
                cur_batch_num = test_feeder.size - cur_batch * FLAGS.batch_size 
            for i in range(cur_batch_num):
                indexs.append(cur_batch * FLAGS.batch_size + i) 
            test_inputs,test_seq_len,test_labels=test_feeder.input_index_generate_batch(indexs)
            cur_labels = [test_feeder.labels[i] for i in indexs]
            test_feed={g.inputs: test_inputs,
                      g.labels: test_labels,
                      g.seq_len: np.array([g.cnn_time]*test_inputs.shape[0])}
            dense_decoded= sess.run(g.dense_decoded,test_feed)
            acc = utils.accuracy_calculation(cur_labels,dense_decoded,ignore_value=-1,isPrint=False)
            acc_all.append(acc)
        print("$$$$$$$$$$$$$$$$$ ACC is :",acc_all,"$$$$$$$$$$$$$$$$$")
        print("avg_acc:",np.array(acc_all).mean()) 

if __name__ == '__main__':
    acc()
