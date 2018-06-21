import model
import utils
import time
import tensorflow as tf
import numpy as np
import os
import logging,datetime

FLAGS=utils.FLAGS
Flag_Isserver = True
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new2'
#data_dir = pre_data_dir + '/test_data/train_words'
#text_dir = pre_data_dir + '/test_data/train_words_gt.txt'
data_dir = pre_data_dir + '/test_data/val_words'
text_dir = pre_data_dir + '/test_data/val_words_gt.txt'


def test(val_dir=data_dir,val_text_dir=text_dir):
    g = model.Graph(is_training=True)
    print('loading validation data, please wait---------------------','end= ')
    val_feeder=utils.DataIterator2(data_dir=val_dir, text_dir=val_text_dir)
    print('***************get image: ',val_feeder.size)

    num_val_samples=val_feeder.size
    num_val_per_epoch=int(num_val_samples/FLAGS.batch_size)
    
    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
    config.gpu_options.allow_growth = True  
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore') 
        
        print('=============================begin testing=============================')
        if True:
            if True:
                if True:
                    acc_avg = 0.0
                    for cur_batch_cv in range(num_val_per_epoch):
                        print(num_val_per_epoch)
                        index_cv = []
                        for i in range(FLAGS.batch_size):
                            index_cv.append(cur_batch_cv*FLAGS.batch_size + i)
                        #print("index_cv",index_cv)
                        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch(index_cv)
                        val_feed={g.inputs: val_inputs,
                            g.labels: val_labels, 
                            g.keep_prob_cv: 1}
                        predict_word_index, lr = sess.run([g.logits,g.learning_rate], val_feed)
                        print(val_labels[0], predict_word_index[0])
                        acc = utils.compute_acc(val_labels, predict_word_index)
                        acc_avg += acc
                    acc_avg = acc_avg / num_val_per_epoch
                    print("acc",acc_avg)
                    


if __name__ == '__main__':
    if Flag_Isserver:
        test()
    else:
        test(train_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train', train_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt',val_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train', val_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt')
