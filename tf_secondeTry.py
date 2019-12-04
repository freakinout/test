from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import os
import create_imgs_fromfont_class
import time

IMG_WIDTH=30
IMG_HEIGHT=30
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
NUM_EPOCHS = 5
FOLDER_PROPERTY_SEPERATOR = "_"
file_ending = ".bmp"

conv_layers = [1,2,3]
conv_depths = [64,128,256]
conv_sizes  = [3,4,5]
pool_sizes  = [2,3]
dense_layers= [0,1,2]
dense_sizes = [64,128,256]


script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\imgs\\"
model_path = script_path + "\\mymodels\\letter_recognition\\"

dataset_size = 0
letter_index_list = []
CATEGORIES = []
CATEGORIE_LABELS = []
i= 0
for folder in os.listdir(img_path):
  a = folder.split(FOLDER_PROPERTY_SEPERATOR)
  for i in range(1,len(a)-1):
    if not a[i][:1] in CATEGORIE_LABELS:
      CATEGORIE_LABELS.append(a[i][:1])
  CATEGORIES.append(a)


def get_dirs(cat_vals):
  dirs = []
  for i in CATEGORIES:
    found = True
    for j in range(len(cat_vals)):
      found = found and ("{}{}".format(CATEGORIE_LABELS[j],cat_vals[j]) in i or len(cat_vals[j])==0)
    if found:  dirs.append(FOLDER_PROPERTY_SEPERATOR.join(i) )
  return dirs

def get_label_from_file(directory, as_tensor=False):
  f = open(img_path+directory+"\\labels.txt")
  label_file = f.readlines()
  f.close()
  labels = np.asarray(label_file)
  if as_tensor: labels = tf.data.Dataset.from_tensor_slices(labels)
  return labels

def get_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_bmp(img, channels=0)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
  return img

def process_labels(label_num,file_name,directory):
  path = tf.strings.join([img_path,directory,"\\",file_name,file_ending])
  img = get_img(path)
  return img, label_num

def preparer_data(directory, first_run=False):
  global letter_index_list
  global dataset_size
  label_file = tf.data.TextLineDataset(img_path+directory+"\\labels.txt")
  l_lines = []

  for l_line in label_file:
    q = tf.strings.split(l_line,";")
    if len(q)>2:
      h = q[0] # plz find a nicer way
      for t in range(len(q)-1):
          if t==0: h=q[t]
          else: h = tf.strings.join([h,q[t]],separator=";")
    else: h = q[0]

    if not h in letter_index_list: 
      letter_index_list.append(h)
    
    for j in range(len(letter_index_list)):
      if letter_index_list[j] == h:
        h = str(j)
        break
    q = tf.stack([h,q[-1]])
    l_lines.append(q)
  dataset_size += len(l_lines)
  l_lines = tf.data.Dataset.from_tensor_slices(l_lines)
  l_lines = l_lines.map(lambda x: (x[0],x[1], directory))
  dataset = l_lines.map(process_labels)

  return dataset

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, shuffeling = True):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  if shuffeling: ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  ds = ds.repeat()
  #ds = ds.batch(1000)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def ds_from_property(property_list):
  props = []
  for c in CATEGORIE_LABELS:
    found_cat = False
    for p in property_list:
      if p[0] == c: 
        props.append(p[1])
        found_cat = True
        break
    if not found_cat: props.append("")
  dirs = get_dirs(props)
  ds = preparer_data(dirs[0])
  for d in dirs[1:]:
    ds = ds.concatenate(preparer_data(d))
  return ds

def make_training(ds):
  #ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  image, label = ds
  return image, label

def preparer_unmapped_data(directory, first_run=False):
  global letter_index_list
  label_file = tf.data.TextLineDataset(img_path+directory+"\\labels.txt")
  l_lines = []
  imgs = []

  for l_line in label_file:
    q = tf.strings.split(l_line,";")
    if len(q)>2:
      h = q[0] # plz find a nicer way
      for t in range(len(q)-1):
          if t==0: h=q[t]
          else: h = tf.strings.join([h,q[t]],separator=";")
    else: h = q[0]

    if not h in letter_index_list: # filling out Letter Index
      letter_index_list.append(h)
    
    for j in range(len(letter_index_list)):
      if letter_index_list[j] == h:
        h = str(j)
        break
    #q = tf.stack([h,q[-1]])
    l_lines.append(q[-1])
    path = tf.strings.join([img_path,directory,"\\",h,file_ending])
    imgs.append(get_img(path))

  l_lines = np.asarray(l_lines)
  imgs    = np.asarray(imgs)

  return imgs, l_lines

def ds_unmapped_from_property(property_list):
  props = []
  for c in CATEGORIE_LABELS:
    found_cat = False
    for p in property_list:
      if p[0] == c: 
        props.append(p[1])
        found_cat = True
        break
    if not found_cat: props.append("")
  dirs = get_dirs(props)
  ds = preparer_data(dirs[0])
  for i in len(ds):
    for d in dirs[1:]:
      ds[i] = ds[i].concatenate(preparer_data(d))
  return ds

# currently working with 1 folder

ds_propertys = [("p","0"),("d","1.0"),("r","0"),("c","1.0")]
ds = ds_from_property(ds_propertys)
"""
dirs = get_dirs(["0","1.0","0","1.0"])
ds = preparer_data(dirs[0])
"""
for image, label in ds.take(1):
  print("Image shape: ", image.numpy().shape)
  #print(image.numpy())
  print("Label: ", label.numpy()) 


ds = ds.repeat()
ds = ds.shuffle(buffer_size=1000)
ds_iter = iter(ds)

def generator(ds_iter, bs= BATCH_SIZE):
  
  while True:
    images = np.zeros((bs,IMG_WIDTH,IMG_HEIGHT,1))
    labels = np.zeros(bs)
    for j in range(bs):
      i , l = next(ds_iter)
      images[j] = np.asarray(i)
      labels[j] = l
    yield (images, labels)


train_ds = generator(ds_iter, BATCH_SIZE)

model_name = "letter_reading_{}_cnn_128-4x2_{}".format(len(letter_index_list),int(time.time()))

tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(model_path,model_name))

# Keras Model
model = keras.models.Sequential()

model.add(Conv2D(128,(4,4), input_shape=(30,30,1)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Activation("relu"))

model.add(Conv2D(128,(4,4)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(94, activation='softmax'))

model.compile(optimizer='adam',#keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])


#model.fit(images, labels, batch_size=64, epochs=NUM_EPOCHS, shuffle=True, validation_split=0.2, callbacks=[tensorboard])

H = model.fit_generator(
  train_ds,
  steps_per_epoch = dataset_size // BATCH_SIZE,
  epochs = NUM_EPOCHS
)

model.save(model_path+model_name+".model")

tf.io.write_file(model_path+"index_list.txt", tf.strings.join(letter_index_list,separator="\n"))




















"""
list_ds = ds_from_property([("p","0"),("d","1.0"),("r","0"),("c","1.0")])
"""

"""for d in dirs[1:]:
  print(d)
  ds = ds.concatenate(preparer_data(d))"""

"""for image, label in ds.take(5):
  print("Image shape: ", image.numpy().shape)
  #print(image.numpy())
  print("Label: ", label.numpy()) 
"""