from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd
import numpy as np
#import IPython.display as display
#from PIL import Image
import matplotlib.pyplot as plt
import os
#import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

script_path = os.path.dirname(os.path.realpath(__file__))+"\\"
os.chdir(script_path)
img_path = script_path+"letterimgs\\"

#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

f = open("labels.txt")
labels = f.readlines()
f.close()

la = tf.data.TextLineDataset(script_path+"labels.txt")



#for i in range(ord("a"),ord("z")+1): CLASS_NAMES.append(i)

for i in range(len(labels)): labels[i] = labels[i].split(";")[0]
h = []
for i in range(ord("a"),ord("z")+1): h.append(chr(i))
h = np.array(h)

def process_file(la):
    #num = int((path.basename(file_path)).split(".")[0])
    p = tf.strings.split(la,";")
    label = h==p[0]
    img = tf.io.read_file(tf.strings.join([img_path,p[1],".bmp"]))
    img = tf.image.decode_bmp(img,channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    #resize already done
    return img, label

def test(la):
    p = tf.strings.split(la,";")
    label = h==p[0]
    return label


list_ds = tf.data.Dataset.range(0,10)

labeled_ds = la.map(process_file, num_parallel_calls=AUTOTUNE)

for  image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

#print(x_test)
#print("prepare files")


