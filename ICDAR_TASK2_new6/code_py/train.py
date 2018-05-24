import model
import utils
import time
import tensorflow as tf
import numpy as np
import os
import logging,datetime

FLAGS=utils.FLAGS
Flag_Isserver = True
logger = logging.getLogger('Traing for ocr using LSTM+CTC')
logger.setLevel(logging.INFO)
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new2'
train_keep_prob_fc = 0.5
train_keep_prob_cv1 = 0.5
train_keep_prob_cv2 = 0.5
train_keep_prob_cv3 = 0.5
train_keep_prob_cv4 = 0.5
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
    config.gpu_options.allow_growth = True 
    #config.gpu_options.per_process_gpu_memory_fraction = 0.6   
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        g.graph.finalize()
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        if FLAGS.restore:
            print("restore is true")
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
        val_feed={g.inputs: val_inputs,
                  g.labels: val_labels,
                  g.seq_len: np.array([g.cnn_time]*val_inputs.shape[0]),
                  g.keep_prob_fc: 1,
                  g.keep_prob_cv1: 1,
                  g.keep_prob_cv2: 1,
                  g.keep_prob_cv3: 1,
                  g.keep_prob_cv4: 1}
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch+1)%100==0:
                    print("cur_epoch====",cur_epoch,"cur_batch----",cur_batch,"g_step****",step,"cost",batch_cost)
                batch_time = time.time()
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                feed={g.inputs: batch_inputs,
                        g.labels:batch_labels,
                        g.seq_len:np.array([g.cnn_time]*batch_inputs.shape[0]),
                        g.keep_prob_fc: train_keep_prob_fc,
                        g.keep_prob_cv1 : train_keep_prob_cv1,
                        g.keep_prob_cv2 : train_keep_prob_cv2,
                        g.keep_prob_cv3 : train_keep_prob_cv3,
                        g.keep_prob_cv4 : train_keep_prob_cv4}

                summary_str, batch_cost,step,_ = sess.run([g.merged_summay,g.cost,g.global_step,g.optimizer],feed)
                train_cost+=batch_cost*FLAGS.batch_size
                train_writer.add_summary(summary_str,step)
                if step%FLAGS.save_steps == 0:
                    print("save checkpoint", step)
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}',format(step))
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)
                if step%FLAGS.validation_steps == 0:
                    dense_decoded,lastbatch_err,lr = sess.run([g.dense_decoded,g.lerr,g.learning_rate],val_feed)
                    acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
                    avg_train_cost=train_cost/((cur_batch+1)*FLAGS.batch_size)
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{}  step==={}, Epoch {}/{}, accuracy = {:.3f},avg_train_cost = {:.3f}, lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}\n"
                    print(log.format(now.month,now.day,now.hour,now.minute,now.second,step,
                        cur_epoch+1,FLAGS.num_epochs,acc,avg_train_cost,lastbatch_err,time.time()-start_time,lr))
                    if Flag_Isserver:
                        f=open('../log/acc/acc.txt',mode="a")
                        f.write(log.format(now.month,now.day,now.hour,now.minute,now.second,step,
                            cur_epoch+1,FLAGS.num_epochs,acc,avg_train_cost,lastbatch_err,time.time()-start_time,lr))
                        f.close()
if __name__ == '__main__':
    if Flag_Isserver:
        train(train_dir= pre_data_dir + '/train_data/train_words', val_dir=pre_data_dir + '/train_data/train_words_small', train_text_dir=pre_data_dir + '/train_data/train_words_gt.txt',val_text_dir=pre_data_dir + '/train_data/train_words_gt.txt')
    else:
        train(train_dir='C:/Users/jieyang/Documents/GitHub/STN_CNN_LSTM_CTC_TensorFlow/train/svt/train1', val_dir='C:/Users/jieyang/Documents/GitHub/STN_CNN_LSTM_CTC_TensorFlow/train/svt/test')
    #
