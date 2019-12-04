from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
import os
import create_imgs_fromfont_class

TF_PICTURE_SIZE = [30,30]
CREATOR_PICTURE_SIZE = [42,60]
FONTSIZES = [24,22,20,18]
LETTER_ASCII = range(65,91)#(33,126)
img_rel_path = "\\tmp_imgs\\"
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_CREATION_MULTIPLIER = 10
BATCH_SPLITTER = 2
TRAINING_REPS = 10
EPOCHS_COUNT = 8

helper=[]
for i in range(BATCH_CREATION_MULTIPLIER):
    helper += FONTSIZES
FONTSIZES = helper
BATCH_SIZE = len(FONTSIZES)*len(LETTER_ASCII)
letter_index_list = []


script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

if img_rel_path[:1]!="\\" and img_rel_path[:1]!="/": img_rel_path = "\\" + img_rel_path
if img_rel_path[-1:]!="\\" and img_rel_path[-1:]!="/": img_rel_path = img_rel_path + "\\"
img_path = script_path+img_rel_path

image_creator = create_imgs_fromfont_class.img_creation(
    position_variance=0,
    distortion_amp=0,
    use_edge_finder=False, #Looses info for training
    rotation_amp=0,
    inverse_colors=True,
    contrast=1,
    letters_ascii=LETTER_ASCII,
    fontsizes=FONTSIZES,
    picture_size=CREATOR_PICTURE_SIZE,
    output_size=TF_PICTURE_SIZE
    )



def preparer_data(data):
    global letter_index_list
    labels = np.zeros(len(data))
    files = np.zeros((len(data),TF_PICTURE_SIZE[0],TF_PICTURE_SIZE[1]))
    for i in range(data.shape[0]):
        if not (data[i,1] in letter_index_list): 
            letter_index_list.append(data[i,1])
        for q in range(len(letter_index_list)):
            if letter_index_list[q] == data[i,1]: labels[i] = q
        files[i] = data[i,0]
    
    return files, labels

def map_fkt(f,l):
    f = tf.reshape(f,(TF_PICTURE_SIZE[0],TF_PICTURE_SIZE[1],1))
    return tf.image.convert_image_dtype(f,tf.float32), l


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, shuffeling = True):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  if shuffeling: ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  ds = ds.repeat()
  ds = ds.batch(round(BATCH_SIZE/BATCH_SPLITTER))

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

def make_ds(forTesting = False):
    ds = image_creator.make_alphabeth(
        save_to_files=False,
        one_font_mode=False
    )

    ds = tf.data.Dataset.from_tensor_slices(preparer_data(ds))
    ds = ds.map(map_fkt, num_parallel_calls=AUTOTUNE)
    ds = prepare_for_training(ds,shuffeling=forTesting)

    return ds

# Keras Model
model = keras.models.Sequential()

model.add(Conv2D(128,(8,8), input_shape=(30,30,1)))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Activation("relu"))

model.add(Conv2D(128,(6,6)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))



model.add(Dense(len(LETTER_ASCII), activation='softmax'))

"""
inputs = keras.Input(shape=(30,30,1), name='digits')
x = keras.layers.Conv2D(64,(8,8),activation='relu')(inputs)
x = keras.layers.MaxPool2D(pool_size=(4,4))(x)
x = keras.layers.Conv2D(128,(8,8),activation='relu')(x)
x = keras.layers.MaxPool2D(pool_size=(4,4))(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu', name='dense_1')(x)
outputs = keras.layers.Dense(len(LETTER_ASCII), activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)"""
model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # List of metrics to monitor
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

image_val, label_val = next(iter(make_ds()))

for i in range(TRAINING_REPS):
    image_batch, label_batch = next(iter(make_ds()))
    if i==2: image_creator.posvar = 5
    if i==4: image_creator.distortion_amp = 2
    if i==6: image_creator.rotation_amp = 40
    print('# Fit model on training data Repition: '+str(i))
    history = model.fit(image_batch,
                        label_batch,
                        epochs=EPOCHS_COUNT,
                        validation_data=(image_val, label_val)
                        )


    # Check History if model worth saving
    model.save(img_path+"myModel"+str(i)+".model")

