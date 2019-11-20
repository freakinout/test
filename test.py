from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_WIDTH=30
IMG_HEIGHT=30
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=5


script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\letterimgs\\"

f = open(img_path+"labels.txt")
labels = f.readlines()
#labels = labels[0] 
f.close()



"""for i in range(len(labels)): 
  q = labels[i].split(";")
  labels[i] = [ord(q[0]),q[1]]
"""
labels = np.asarray(labels)
labels = tf.data.Dataset.from_tensor_slices(labels)

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
  return p#==h

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


def process_labels(label_arr):
  helper = tf.strings.split(label_arr,sep=";")
  l = tf.math.subtract(tf.strings.unicode_decode(tf.gather(helper,0),input_encoding='UTF-8'),tf.strings.unicode_decode("a",input_encoding='UTF-8'))
  img = tf.strings.join([img_path,tf.strings.strip(tf.gather(helper,1)),".bmp"])
  img = tf.io.read_file(img)
  img = tf.image.decode_bmp(img, channels=0)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) , l


list_ds = labels.map(process_labels, num_parallel_calls=AUTOTUNE)

image_batch, label_batch = next(iter(list_ds))

model = tf.keras.models.load_model(script_path+"\\letterimgs_50k_arial\\"+"myModel.model")

m = model.predict(image_batch)