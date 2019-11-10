from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
#import IPython.display as display
#from PIL import Image
import matplotlib.pyplot as plt
import os
#import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\letterimgs\\"



f = open("labels.txt")
labels = f.readlines()
f.close()
for i in range(len(labels)): labels[i] = labels[i].split(";")[0]
h = []
for i in range(ord("a"),ord("z")+1): h.append(chr(i))
h = np.array(h)

q = [h==labels[0]]
#for a in labels: q.append(h==a)

print(q)
print(h)





exp = tf.data.experimental.make_csv_dataset(
    script_path+"labels.txt",
    batch_size=10,
    header=False,
    field_delim=';',
    column_names=['letter','num'],
    select_columns=['letter']
)
