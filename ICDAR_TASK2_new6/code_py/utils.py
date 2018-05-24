import os,sys
import numpy as np
import tensorflow as tf
import random
import cv2,time
from skimage.util import random_noise
from skimage import transform
from tensorflow.python.client import device_lib
import re

channel = 1
image_width=160
image_height=32
num_features = image_height*channel
SPACE_INDEX=0
SPACE_TOKEN=''
aug_rate=100
maxPrintLen = 18
pre_dir = '/home/sjhbxs/checkout/ICDAR_task2/ICDAR_TASK2_new6' 
tf.app.flags.DEFINE_boolean('isSavePrediction',True, 'save test prediction')
tf.app.flags.DEFINE_boolean('Use_CRNN',True, 'use Densenet or CRNN')
tf.app.flags.DEFINE_boolean('restore',True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', pre_dir + '/checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
tf.app.flags.DEFINE_integer('num_layers', 2, 'number of layer')
tf.app.flags.DEFINE_integer('num_hidden', 256, 'number of hidden')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 256, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 400, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 1000, 'the step to validation')
tf.app.flags.DEFINE_float('decay_rate', 0.99, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
FLAGS=tf.app.flags.FLAGS

charset='! "#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]_`abcdefghijklmnopqrstuvwxyz~¡¢£©°É•€★'
num_classes=len(charset)+2

encode_maps={}
decode_maps={}
for i,char in enumerate(charset,1):
    encode_maps[char]=i
    decode_maps[i]=char
encode_maps[SPACE_TOKEN]=SPACE_INDEX
decode_maps[SPACE_INDEX]=SPACE_TOKEN


class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        self.image = []
        self.labels=[]
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                try:
                    image_name = os.path.join(root,file_path)
                    im = cv2.imread(image_name,0)#/255.#read the gray image
                    img = cv2.resize(im, (image_width, image_height))
                    img = img.swapaxes(0, 1)
                    code = file_path.split('_')[1]
                    code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]   
                    self.labels.append(code)
                    self.image.append(np.array(img[:,:,np.newaxis]))
                    self.image_names.append(file_path)
                except:
                    print("!!!!!!!!!!!",root,"---",file_path,"=====",flag_a)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self,indexs):
        labels=[]
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    def input_index_generate_batch(self,index=None):
        if index:
            image_batch=[self.image[i] for i in index]
            label_batch=[self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch=self.image
            label_batch=self.labels

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences,lengths
        batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels

class DataIterator2:
    def __init__(self, data_dir,text_dir):
        num_string = {}
        f = open(text_dir,"r")
        line = f.readline()  
        while line:  
            contant_line = str(line)
            if ',' in contant_line:
                 num = contant_line.split(',')[0]
                 string_line = contant_line.split(',')[1]
            else:
                line = f.readline()  
                continue
            if string_line[0] == '|':
                string_line = contant_line.split('|')[1]
            string_line = re.sub('[\r\n\t]', '', string_line)
            num_string[num] = str(string_line)
            line = f.readline()  
        self.image_names = []
        self.image = []
        self.labels=[]
        self.total_pic_read = 0
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                try:
                    self.total_pic_read += 1
                    image_name = os.path.join(root,file_path)
                    im = cv2.imread(image_name,0)#/255.#read the gray image
                    img = cv2.resize(im, (image_width, image_height))
                    img = img.swapaxes(0, 1)
                    num_from_file_path = file_path.split('.')[0]
                    code = num_string[str(num_from_file_path)]
                    code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]   
                    if len(code) == 0 or len(code) > 35:
                        print("len is 0 or too long",num_from_file_path)
                        continue
                    self.labels.append(code)
                    self.image.append(np.array(img[:,:,np.newaxis]))
                    self.image_names.append(image_name)
                except:
                    print("!!!!!!!!!!!",root,"---",file_path,"=====",flag_a)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self,indexs):
        labels=[]
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    def input_index_generate_batch(self,index=None):
        if index:
            image_batch=[self.image[i] for i in index]
            label_batch=[self.labels[i] for i in index]
        else:
            # get the whole data as input
            image_batch=self.image
            label_batch=self.labels

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences,lengths
        batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels

class DataIterator3:
    def __init__(self, data_dir):
        self.image_num = []
        self.image = []
        self.total_pic_read = 0
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                try:
                    self.total_pic_read += 1
                    image_name = os.path.join(root,file_path)
                    im = cv2.imread(image_name,0)#/255.#read the gray image
                    img = cv2.resize(im, (image_width, image_height))
                    img = img.swapaxes(0, 1)
                    num_from_file_path = file_path.split('.')[0]
                    self.image.append(np.array(img[:,:,np.newaxis]))
                    self.image_num.append(num_from_file_path)
                except:
                    print("!!!!!!!!!!!",root,"---",file_path,"=====",flag_a)


    @property
    def size(self):
        return len(self.image_num)

    def the_label(self,indexs):
        labels=[]
        for i in indexs:
            labels.append(self.labels[i])
        return labels

    def input_index_generate_batch(self,index=None):
        if index:
            image_batch=[self.image[i] for i in index]
            num_batch=[self.image_num[i] for i in index]
        else:
            # get the whole data as input
            image_batch=self.image
            label_batch=self.image_num

        def get_input_lens(sequences):
            lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
            return sequences,lengths
        batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))
        
        return batch_inputs,num_batch

def accuracy_calculation(original_seq,decoded_seq,ignore_value=-1,isPrint = True):
    if  len(original_seq)!=len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i,origin_label in enumerate(original_seq):
        decoded_label  = [j for j in decoded_seq[i] if j!=ignore_value]
        if isPrint and i<maxPrintLen:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i,origin_label,decoded_label))
        if origin_label == decoded_label: count+=1
    return count*1.0/len(original_seq)

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(0,len(seq),1)))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

def pad_input_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.
        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def decode_function1(index_list):
    index_list_new = []
    for e in index_list:
        if e != -1 and e < len(charset):
            index_list_new.append(e)
    code = [SPACE_TOKEN if index_list_new == SPACE_INDEX else decode_maps[c] for c in list(index_list_new)]
    code = ''.join(code)
    print(code)
    return code

def decode_function(index_list):
    index_list_new = []
    for e in index_list:
        if e != -1:
            index_list_new.append(e)
    code = [SPACE_TOKEN if index_list_new == SPACE_INDEX else decode_maps[c] for c in list(index_list_new)]
    code = ''.join(code)
    return code
