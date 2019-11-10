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



la = tf.io.read_file("labels.txt")
p = tf.strings.split(la,";")

