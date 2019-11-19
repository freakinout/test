from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint

IMG_WIDTH=30
IMG_HEIGHT=30
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE=50


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
  return p#==h

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label


list_ds = tf.data.Dataset.list_files(tf.strings.join([img_path,'*.bmp']))

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

""" for image, label in labeled_ds.take(3):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy()) """


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

train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

#print("####")
print(label_batch.shape)

inputs = keras.Input(shape=(30,30,1), name='digits')
x = layers.Flatten()(inputs)
x = layers.Dense(64, activation='relu', name='dense_1')(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(26, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

#(x_train, y_train) = labeled_ds

#x_train = x_train.reshape(500, 900).astype('float32')

#x_val = x_train[-100:]
#x_train = x_train[:-100]

model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Train the model by slicing the data into "batches"
# of size "batch_size", and repeatedly iterating over
# the entire dataset for a given number of "epochs"
print('# Fit model on training data')
history = model.fit(image_batch,
                    label_batch,
                    #batch_size=50,
                    epochs=3
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    #validation_data=(x_val, y_val)
                    )
