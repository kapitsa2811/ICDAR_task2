import cv2,time,os,re
import tensorflow as tf
import numpy as np
import utils
import model
FLAGS = utils.FLAGS

data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new4/test_data/val_words'
save_dir = '../log/text_save/val_predict.txt'

def predict_func_tem():
    if True:    
        test_feeder=utils.DataIterator3(data_dir=data_dir)
        print(test_feeder.image_num) 
        for cur_batch in range(int(test_feeder.size / FLAGS.batch_size) + 1):
            indexs=[]
            cur_batch_num = FLAGS.batch_size
            if cur_batch == int(test_feeder.size / FLAGS.batch_size):
                cur_batch_num = test_feeder.size - cur_batch * FLAGS.batch_size 
            for i in range(cur_batch_num):
                indexs.append(cur_batch * FLAGS.batch_size + i) 
 
def predict_func():
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
        test_feeder=utils.DataIterator3(data_dir=data_dir)
        f=open(save_dir,mode="a")
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
            test_inputs, num_batch=test_feeder.input_index_generate_batch(indexs)
            test_feed={g.inputs: test_inputs,
                      g.seq_len: np.array([g.cnn_time]*test_inputs.shape[0])}
            dense_decoded= sess.run(g.dense_decoded,test_feed)
            for encode_list, e_num in zip(dense_decoded,num_batch):
                decode_string = utils.decode_function(encode_list)
                # print(e_num,decode_string)
                f.write(e_num + "," + decode_string+"\n")
        f.close()
        print("saved prediction")
    

if __name__ == '__main__':
    predict_func()
