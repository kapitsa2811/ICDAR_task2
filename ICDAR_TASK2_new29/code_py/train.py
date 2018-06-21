import model
import utils
import time
import tensorflow as tf
import numpy as np
import os
import logging,datetime

FLAGS=utils.FLAGS
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new2'


def train(train_dir=None,val_dir=None,train_text_dir=None,val_text_dir=None):
    g = model.Graph(is_training=True)
    print('loading train data, please wait---------------------','end= ')
    train_feeder=utils.DataIterator2(data_dir=train_dir,text_dir=train_text_dir)
    print('***************get image: ',train_feeder.size)
    print('loading validation data, please wait---------------------','end= ')
    val_feeder=utils.DataIterator2(data_dir=val_dir, text_dir=val_text_dir)
    print('***************get image: ',val_feeder.size)

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)
    num_val_samples=val_feeder.size
    num_val_per_epoch=int(num_val_samples/FLAGS.batch_size)
    
    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)  
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        g.graph.finalize()
        if FLAGS.restore:
            print("restore is true")
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        val_inputs,val_seq_len,val_labels,val_labels_len=val_feeder.input_index_generate_batch()
        val_feed={g.inputs: val_inputs,
                  g.y_: val_labels_len,
                  g.keep_prob_fc: 1,
                  g.keep_prob_cv1: 1}
        
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            for cur_batch in range(num_batches_per_epoch):
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels,batch_labels_len=train_feeder.input_index_generate_batch(indexs)
                feed={g.inputs: batch_inputs,
                        g.y_:batch_labels_len,
                        g.keep_prob_fc: FLAGS.train_keep_prob_fc,
                        g.keep_prob_cv1 : FLAGS.train_keep_prob_cv}
                _,step = sess.run([g.train_step,g.global_step],feed)
                if (step+1) % FLAGS.validation_steps == 0:
                    train_accuracy = sess.run([g.accuracy],feed)
                    val_accuracy  = sess.run([g.accuracy],val_feed)
                    print("===step", step, "train_acc:",train_accuracy,"val_acc",val_accuracy)

                if (step+1) % FLAGS.save_steps == 0:
                    print("save checkpoint", step)
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)

if __name__ == '__main__':
    if FLAGS.Isserver:
        train(train_dir= pre_data_dir + '/train_data/train_words', val_dir=pre_data_dir + '/train_data/train_words_small', train_text_dir=pre_data_dir + '/train_data/train_words_gt.txt',val_text_dir=pre_data_dir + '/train_data/train_words_gt.txt')
    else:
        train(train_dir='C:/Users/jieyang/Desktop/Data Science/ICDAR/COCO/TASK2/data/small_train', val_dir='C:/Users/jieyang/Desktop/Data Science/ICDAR/COCO/TASK2/data/small_train',train_text_dir = 'C:/Users/jieyang/Desktop/Data Science/ICDAR/COCO/TASK2/data/train_words_gt.txt',val_text_dir = 'C:/Users/jieyang/Desktop/Data Science/ICDAR/COCO/TASK2/data/train_words_gt.txt')

