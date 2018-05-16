import os,sys
import numpy as np
import tensorflow as tf
import random
import cv2,time
from skimage.util import random_noise
from skimage import transform
from tensorflow.python.client import device_lib
pre_data_dir = '/home/sjhbxs/Data/data_coco_task2/ICDAR_TASK2_new3'
data_dir = pre_data_dir + "/train_data/train_words/"
image_width=160
image_height=32
total_pic_read = 0
if True:
    if True:
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                if total_pic_read > 20:
                    break
                if True:
                    total_pic_read += 1
                    image_name = os.path.join(root,file_path)
                    im = cv2.imread(image_name,0)#/255.#read the gray image
                    width, hight = im.shape
                    if hight > width:
                        print("In Utils, hight > width, so I am wrong")
                    img = cv2.resize(im, (image_height, image_width))
                    save_dir_name = "./pic_test/" + str(total_pic_read) + ".jpg"
                    save_dir_name_ori = "./pic_test/ori" + str(total_pic_read) + ".jpg"

                    cv2.imwrite(save_dir_name,img)
                    cv2.imwrite(save_dir_name_ori, im)
                    img_np = np.array(img[:,:,np.newaxis])
                    print(img_np.shape)
                    
