import os,sys
import numpy as np
import tensorflow as tf
import random
import cv2,time
from skimage.util import random_noise
from skimage import transform
from tensorflow.python.client import device_lib
import re
#10 digit + blank + space

data_dir='/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new4/train_data/train_words_argu'
count_file = 0
images = []
for root, sub_folder, file_list in os.walk(data_dir):
    for file_path in file_list:
        if count_file > 20000:
            break
        count_file += 1
        if count_file % 1000 == 0:
            print("count_file", count_file, file_path)
        image_name = root + "/" + file_path
        im = cv2.imread(image_name,1)#/255.#read the gray image
        ''' img = cv2.resize(im, (120, 30))
        img = img.swapaxes(0, 1)
        num_from_file_path = file_path.split('.')[0]
        num_from_file_path = num_from_file_path.split('_')[1]'''
        #images.append(img)
