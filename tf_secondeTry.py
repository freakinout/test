from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint

IMG_WIDTH=30
IMG_HEIGHT=30
AUTOTUNE = tf.data.experimental.AUTOTUNE



script_path = os.path.dirname(os.path.realpath(__file__))+"\\"
os.chdir(script_path)
img_path = script_path+"letterimgs\\"


f = open(img_path+"labels.txt")
labels = f.readlines()
f.close()

for i in range(len(labels)): labels[i] = (labels[i].split(";")[0])

h = []
for i in range(ord("a"),ord("z")+1): h.append(chr(i))
h = np.asarray(h)


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_bmp(img, channels=0)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def get_label(file_path):
  p = tf.strings.split(file_path,"\\")[-1]
  p = tf.strings.split(p,".")[0]
  p = tf.gather(labels,tf.strings.to_number(p,out_type=tf.dtypes.int32))
  #p = tf.strings.split(p,";")[0]
  return h==p

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


list_ds = tf.data.Dataset.list_files(tf.strings.join([img_path,'*.bmp']))

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(3):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


  