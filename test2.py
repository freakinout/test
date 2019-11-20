#from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image, ImageDraw, ImageFont
#import tensorflow as tf
import numpy as np
#import IPython.display as display
#from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from typing import List
#import pathlib
#AUTOTUNE = tf.data.experimental.AUTOTUNE

import create_imgs_fromfont_class
import numpy as np

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