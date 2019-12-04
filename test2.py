#from __future__ import absolute_import, division, print_function, unicode_literals
#from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
#import numpy as np
#import IPython.display as display
#from PIL import Image
#import matplotlib.pyplot as plt
#import random
#import os
from typing import List
#import pathlib
#AUTOTUNE = tf.data.experimental.AUTOTUNE

import create_imgs_fromfont_class
import numpy as np

def mapping(ds):
    return np.zeros((30,30)), 1

ds = tf.data.Dataset.from_tensor_slices(np.zeros((30,30)))

ds = ds.map(mapping)

ds = tf.data.Dataset.zip(*ds)

"""script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\imgs\\"

import shutil

def myrev(t):
    return(tf.reverse(t,[-1]))

def rev2(t):
    #print(tf.DType.is_integer(t))
    try:
        t = tf.reverse(t,[-1])
        return t
    except:
        h=[]
        for i in t:
            h.append([rev2(i)])
        return tf.ragged.stack(h,axis=0)

txt = tf.strings.unicode_decode("T_TEST_STRING",input_encoding='UTF-8')
txt = tf.strings.unicode_encode(txt,output_encoding='UTF-8')
print(txt)
print(tf.reverse(txt,axis=0))
txt = tf.strings.split(txt,"_")
print(txt)
txt = tf.strings.unicode_decode(txt,input_encoding='UTF-8')
print(txt)
#print(txt.numpy())
txt = rev2(txt)
#txt = txt.map(myrev)
print(txt)
txt = tf.strings.unicode_encode(txt,output_encoding='UTF-8')
#print(txt.numpy())
print(txt)

txt = "T_TE___ST_STRING"
txt = tf.strings.split(txt,"_")
if len(txt)>2:
    #h=""
    for t in range(len(txt)-1):
        if t==0: h=txt[t]
        else: h = tf.strings.join([h,txt[t]],separator="_")
    txt = [h,txt[-1]]
print(txt)
#shutil.rmtree(img_path, ignore_errors=True)"""
"""
dirs = []
print(os.listdir(img_path))
for i in range(100000):
    if len(os.listdir(img_path))>0: 
        a = os.listdir(img_path)[0]
        #print(a)
        img_path += a+"\\"
        dirs.append(a)
    else:
        print("found end @"+str(i))
        break


helper = dirs
for q in range(len(dirs)):
    img_path = script_path+"/imgs/"+"/".join(helper)+""
    helper.remove(dirs[-q-1])
    shutil.rmtree(img_path)
  """
"""
a = create_imgs_fromfont_class.img_creation(
    position_variance=0,
    distortion_amp=4,
    use_edge_finder=False, #Looses info for training
    edge_finder_border=3
    )



b = a.make_letter("Q","C:/Windows/Fonts/arial.ttf",15)
b = a.rotate_img(b,255,30)
b.save("test_name.bmp")


b = np.array(b)
print(b[0,0])
"""





"""
q = []
for i in range(5000):
    q.append(abs(0.3*random.gauss(0,0.1)))

s=20
v = 0
for j in range(s):
    x = 0
    for i in q:
        if i>=v and i<v+1/s: x+=1
    print("v= ",v," - x= ",x)
    v+=1/s
"""


"""
def edge_finder(img,zerocolor):
    left = len(img[0])
    top = len(img)
    right = 0
    bottom = 0

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i,j]!=zerocolor:
                left = min([left,j])
                top = min([top,i])
                right = max([right,j])
                bottom = max([bottom,i])
    return (left,top,right,bottom)


border=40

fnt = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20 )
bcolor = [255]
if len(bcolor)==1:
    img = Image.new('L', (40,60), color = bcolor[0])
else:
    img = Image.new('RGB', (5, 5), color = tuple(bcolor))
d = ImageDraw.Draw(img)
d.text((10,10), "a" , font=fnt, fill=0)
#print(d)
c = np.array(img)
(left,top,right,bottom) = edge_finder(c,255)
print((left,top,right,bottom))
img = img.crop(box=[left-border,top-border,right+border,bottom+border])
c = np.array(img)
print(len(c))
img.save("test.bmp")
print()
#print(c)"""

"""def reverse_tf_ragged(t): # Currently not needed
    try:
        t = tf.reverse(t,[-1])
        return t
    except:
        h=[]
        for i in t:
            h.append([reverse_tf_ragged(i)])
        return tf.ragged.stack(h,axis=0)
"""