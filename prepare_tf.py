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
BATCH_SIZE = 32


script_path = os.path.dirname(os.path.realpath(__file__))+"\\"
os.chdir(script_path)
img_path = script_path+"letterimgs\\"

#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

f = open(img_path+"labels.txt")
labels = f.readlines()
f.close()

la = tf.data.TextLineDataset(script_path+"labels.txt")



#for i in range(ord("a"),ord("z")+1): CLASS_NAMES.append(i)

for i in range(len(labels)): labels[i] = labels[i].split(";")[0]
h = []
for i in range(ord("a"),ord("z")+1): h.append(chr(i))
h = np.asarray(h)

def process_file(la):
    #num = int((path.basename(file_path)).split(".")[0])
    p = tf.strings.split(la,";")
    label = h==p[0]
    img = tf.io.read_file(tf.strings.join([img_path,p[1],".bmp"]))
    img = tf.io.decode_bmp(img,channels=0)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [len(img[0]), len(img)])
    return img, label

def test(la):
    p = tf.strings.split(la,";")
    label = h==p[0]
    return label


list_ds = tf.data.Dataset.range(0,len(labels))

labeled_ds = la.map(process_file, num_parallel_calls=AUTOTUNE)

for  image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


#train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(labeled_ds))
image_batch = image_batch.numpy().reshape(len(image_batch),len(image_batch[0])*len(image_batch[0,0]))
label_batch = label_batch.numpy().astype('float32')
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(30, 30, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

model.fit(image_batch, label_batch, batch_size=64, epochs=10)


#def not_doing():
