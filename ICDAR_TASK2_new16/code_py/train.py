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


def train(train_dir=None,val_dir=None,train_text_dir=None,val_text_dir=None):
    acc_avg = 0.0
    acc_best = 0.0
    acc_best_step = 0
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
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = 0
            for cur_batch in range(num_batches_per_epoch):
                now = datetime.datetime.now()
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                feed={g.inputs: batch_inputs,
                        g.labels:batch_labels,
                        g.keep_prob: FLAGS.train_keep_prob_cv}

                batch_cost_avg,step,_ ,predict_result= sess.run([g.cost_batch_avg,g.global_step,g.optimizer,g.logits],feed)
                if step%1 == 0:
                    print("cur_epoch====",cur_epoch,"cur_batch----",cur_batch,"g_step****",step,"cost",batch_cost_avg)
                    if False:
                        print("real",batch_labels)
                        print("predict", predict_result)

                if step%FLAGS.validation_steps == 0:
                    acc_avg = 0.0
                    for cur_batch_cv in range(num_val_per_epoch):
                        print(num_val_per_epoch)
                        index_cv = []
                        for i in range(FLAGS.batch_size):
                            index_cv.append(cur_batch_cv*FLAGS.batch_size + i)
                        print("index_cv",index_cv)
                        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch(index_cv)
                        val_feed={g.inputs: val_inputs,
                            g.labels: val_labels, 
                            g.keep_prob: 1}
                        predict_word_index, lr = sess.run([g.logits,g.learning_rate], val_feed)
                        acc = utils.compute_acc(val_labels, predict_word_index)
                        acc_avg += acc
                    acc_avg = acc_avg / num_val_per_epoch
                    if acc_avg - acc_best > 0.00001:
                        acc_best = acc_avg
                        acc_best_step = step

                    print("acc",acc_avg)
                    if Flag_Isserver:
                        f=open('../log/acc/acc.txt',mode="a")
                        log = "{}/{} {}:{}:{}, Epoch {}/{}, step=={}-->cur_acc = {:.3f}, best_step=={}-->best_acc = {:.3f}, lr={:.8f},batch_cost_avg={:.3f}\n"
                        f.write(log.format(now.month,now.day,now.hour,now.minute,now.second,cur_epoch+1,FLAGS.num_epochs,step,acc_avg,acc_best_step,acc_best,lr,batch_cost_avg))
                        f.close()

                if step%FLAGS.save_steps == 0 or acc_avg - acc_best > 0.05:
                    print("save checkpoint", step)
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)


if __name__ == '__main__':
    if Flag_Isserver:
        train(train_dir= pre_data_dir + '/train_data/train_words', val_dir=pre_data_dir + '/train_data/val_words_small', train_text_dir=pre_data_dir + '/train_data/train_words_gt.txt',val_text_dir=pre_data_dir + '/train_data/val_words_gt.txt')
    else:
        train(train_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train', train_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt',val_dir=r'C:/Users/jieyang/Desktop/Python/data/small_train', val_text_dir=r'C:/Users/jieyang/Desktop/Python/data/train_words_gt.txt')
