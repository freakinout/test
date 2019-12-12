from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation, InputLayer
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import create_imgs_fromfont_class
import time
import random

print("Import completed")

small_data_for_testing = True

IMG_WIDTH=30
IMG_HEIGHT=30
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
NUM_EPOCHS = 8
FOLDER_PROPERTY_SEPERATOR = "_"
file_ending = ".bmp"



script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\imgs\\"
model_path = script_path + "\\mymodels\\letter_recognition\\"

dataset_size = 0
val_dataset_size = 0
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

print("Found {} Categories, with {} Labels".format(len(CATEGORIES),len(CATEGORIE_LABELS)))

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

def preparer_data(directory, val_data=False):
  global letter_index_list
  global dataset_size
  global val_dataset_size
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
  if val_data: val_dataset_size += len(l_lines) 
  else:  dataset_size += len(l_lines)
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
  ds = ds.cache()
  ds = ds.repeat()
  ds = ds.shuffle(buffer_size=1000)
  #ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds


def generator(ds_iter, bs= BATCH_SIZE):
  
  while True:
    images = np.zeros((bs,IMG_WIDTH,IMG_HEIGHT,1))
    labels = np.zeros(bs)
    for j in range(bs):
      i , l = next(ds_iter)
      images[j] = np.asarray(i)
      labels[j] = l
    yield (images, labels)
  


def tb_log_writer():
  print("not yet")

print("Preparing data ")

# using just one folder
dirs = get_dirs(["0","1.0","0","1.0"])
#split off validation_data
ds_val = preparer_data( dirs[1], val_data=True).repeat()
ds_val = generator( iter( ds_val) )

if small_data_for_testing:
  ds = preparer_data(dirs[0])
else:
  ds_propertys = [("p","0"),("d","1.0"),("r","0"),("c","1.0")]
  ds = ds_from_property(ds_propertys)


ds_iter = iter(make_training(ds))


train_ds = generator(ds_iter, BATCH_SIZE)



print("Creating models")
conv_layers = [1,2,3]
conv_depths = [64,128,256]
conv_sizes  = [3,4,5]
pool_sizes  = [2,3]
dense_layers= [0,1,2]
dense_sizes = [64,128,256]

output_size = len(letter_index_list)

for conv_layer in conv_layers:
  for conv_depth in conv_depths:
    for conv_size in conv_sizes:
      for pool_size in pool_sizes:
        for dense_layer in dense_layers:
          for dense_size in dense_sizes:

            model_name = "letters-{}_{}cnn_{}x{}sqrd_pool-{}_dense{}x{}_{}".format(
              output_size,
              conv_layer, conv_depth, conv_size,
              pool_size,
              dense_layer, dense_size,
              int(time.time())
              )

            tensorboard = TensorBoard(log_dir='{}logs\\{}'.format(model_path,model_name))

            # Keras Model
            model = keras.models.Sequential()

            model.add(InputLayer(input_shape=(30,30,1)))

            for i in range(conv_layer):
              model.add(Conv2D(conv_depth,(conv_size,conv_size)))
              model.add(MaxPool2D(pool_size=(pool_size,pool_size)))
              model.add(Activation("relu"))


            model.add(Flatten())

            for i in range(dense_layer):
              model.add(Dense(dense_size))
              model.add(Activation("relu"))

            model.add(Dense(output_size, activation='softmax'))

            model.compile(optimizer='adam',
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=[keras.metrics.SparseCategoricalAccuracy()])


            #model.fit(images, labels, batch_size=64, epochs=NUM_EPOCHS, shuffle=True, validation_split=0.2, callbacks=[tensorboard])

            H = model.fit_generator(
              train_ds,
              steps_per_epoch = dataset_size // BATCH_SIZE,
              validation_steps = val_dataset_size // BATCH_SIZE,
              validation_data = ds_val,
              epochs = NUM_EPOCHS,
              callbacks=[tensorboard]
            )

            model.save(model_path+model_name+".model")
            tf.io.write_file(model_path+model_name+".model\\index_list.txt", tf.strings.join(letter_index_list,separator="\n"))


