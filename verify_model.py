from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import create_imgs_fromfont_class

TF_PICTURE_SIZE = [30,30]
CREATOR_PICTURE_SIZE = [42,60]
FONTSIZES = [21]
LETTER_ASCII = range(33,126)
img_rel_path = "\\tmp_imgs\\"
AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = len(FONTSIZES)*len(LETTER_ASCII)


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




def preparer_data(data, index_list = []):
    labels = np.zeros(len(data))
    files = np.zeros((len(data),TF_PICTURE_SIZE[0],TF_PICTURE_SIZE[1]))
    for i in range(data.shape[0]):
        if not (data[i,1] in index_list): 
            index_list.append(data[i,1])
        for q in range(len(index_list)):
            if index_list[q] == data[i,1]: labels[i] = q
        files[i] = data[i,0]
    
    return (files, labels, np.asarray(index_list))

def map_fkt(f,l):
    return tf.image.convert_image_dtype(f,tf.float32), l


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds


ds = image_creator.make_alphabeth(
    save_to_files=False,
    one_font_mode=False
)

(files, labels, letter_index_list) = preparer_data(ds)

ds = tf.data.Dataset.from_tensor_slices((files,labels))
ds = ds.map(map_fkt, num_parallel_calls=AUTOTUNE)

ds = prepare_for_training(ds)
image_batch, label_batch = next(iter(ds))

model = keras.models.load_model(img_path+"myModel.model")

model.evaluate(x=image_batch,
              y=label_batch
              )