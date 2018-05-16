#import model
import model_transfer
import utils
import time
import tensorflow as tf
import numpy as np
import os
import logging,datetime
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
FLAGS=utils.FLAGS
Flag_Isserver = True
logger = logging.getLogger('Traing for ocr using LSTM+CTC')
logger.setLevel(logging.INFO)
#change_yj
Flage_width = 37
Flage_print_frequency = 50
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new2'

def train(train_dir=None,val_dir=None,train_text_dir=None,val_text_dir=None, pb_file_path = None):
    g = model_transfer.Graph(is_training=True, pb_file_path = pb_file_path)

    print('loading train data, please wait---------------------','end= ')
    train_feeder=utils.DataIterator2(data_dir=train_dir,text_dir=train_text_dir)
    print('***************get image: ',train_feeder.size)
    print('loading validation data, please wait---------------------','end= ')
    val_feeder=utils.DataIterator2(data_dir=val_dir, text_dir=val_text_dir)
    print('***************get image: ',val_feeder.size)
    '''
    print('loading train data, please wait---------------------','end= ')
    train_feeder=utils.DataIterator(data_dir=train_dir)
    print('get image: ',train_feeder.size)
    print('loading validation data, please wait---------------------','end= ')
    val_feeder=utils.DataIterator(data_dir=val_dir)
    print('get image: ',val_feeder.size)
    '''
    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)
    num_val_samples=val_feeder.size
    num_val_per_epoch=int(num_val_samples/FLAGS.batch_size)

    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
    config.gpu_options.allow_growth = True 
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=3)
        if FLAGS.restore:
            print("restore is true")
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))
                
        g.graph.finalize() 
        print('=============================begin training=============================')
        cur_training_step = 0
        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
        val_feed={g.original_pic: val_inputs,
                  g.labels: val_labels,
                  g.seq_len: np.array([Flage_width]*val_inputs.shape[0])}
        
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = 0
            for cur_batch in range(num_batches_per_epoch): 
                cur_training_step += 1
                if  cur_training_step % Flage_print_frequency == 0:
                    print("epochs",cur_epoch, cur_batch)   
                
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                transfer_train_batch_feed={g.original_pic: batch_inputs,
                          g.seq_len: np.array([Flage_width]*batch_inputs.shape[0]),
                          g.labels : batch_labels}
                summary_str, batch_cost,all_step,_ = sess.run([g.merged_summay,g.cost,g.global_step,g.optimizer], transfer_train_batch_feed)
                train_cost += batch_cost*FLAGS.batch_size
                
                if all_step%FLAGS.save_steps == 0:
                    print("**********save checkpoint********** all_step:", all_step)
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=all_step)
                    
                if all_step%FLAGS.validation_steps == 0:
                    print("**********CrossValidation********** all_step:", all_step)
                    dense_decoded,lastbatch_err,lr = sess.run([g.dense_decoded,g.lerr,g.learning_rate],val_feed)
                    acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
                    avg_train_cost=train_cost/((cur_batch+1)*FLAGS.batch_size)
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{}  all_step==={}, Epoch {}/{}, accuracy = {:.3f},avg_train_cost = {:.3f}, lastbatch_err = {:.3f},lr={:.8f}\n"
                    print(log.format(now.month,now.day,now.hour,now.minute,now.second,all_step,
                        cur_epoch+1,FLAGS.num_epochs,acc,avg_train_cost,lastbatch_err,lr))
                    if Flag_Isserver:
                        f=open('../log/acc/acc.txt',mode="a")
                        f.write(log.format(now.month,now.day,now.hour,now.minute,now.second,all_step,
                            cur_epoch+1,FLAGS.num_epochs,acc,avg_train_cost,lastbatch_err,lr))
                        f.close()

if __name__ == '__main__':
    if Flag_Isserver:
        train(train_dir= pre_data_dir + '/train_data/train_words', val_dir=pre_data_dir + '/train_data/train_words_small', train_text_dir=pre_data_dir + '/train_data/train_words_gt.txt',val_text_dir=pre_data_dir + '/train_data/train_words_gt.txt', pb_file_path = "../log/save_pb/rcnn.pb")
    else:
        train(train_dir='C:/Users/jieyang/Documents/GitHub/STN_CNN_LSTM_CTC_TensorFlow/train/svt/train1', val_dir='C:/Users/jieyang/Documents/GitHub/STN_CNN_LSTM_CTC_TensorFlow/train/svt/test')
    print("OK!!!!!!!!!!!!!!!!!!!!!!!!!")
    
