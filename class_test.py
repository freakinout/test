import create_imgs_fromfont_class
import numpy as np
import random
import tensorflow as tf 
import os

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+"\\tmp_imgs\\"

a = create_imgs_fromfont_class.img_creation(
    position_variance=0,
    distortion_amp=0,
    use_edge_finder=False, #Looses info for training
    edge_finder_border=0,
    rotation_amp=0,
    inverse_colors=False,
    contrast=1
    )

c = []
fnts =  a.get_fonts()
for i in range(65,91):
    b = a.make_letter(chr(i),fnts[round(random.randrange(0,len(fnts)))],15)
    b = a.resize_to_output(b)
    b.save(img_path+"test_"+str(i)+".bmp")
    b = np.array(b)
    b = tf.image.convert_image_dtype(b,tf.float32)
    b = tf.reshape(b,(30,30,1))
    c.append(b)
#a.test_font = "arial.ttf"
#b = a.make_alphabeth(True,one_font_mode="arial.ttf")
#print(a.fnts)

c = np.asarray(c)

model = tf.keras.models.load_model(img_path+"myModel9.model")

print(c.shape)

l = model.predict(x=c)

for m in l:
    print(np.argmax(m))

"""
print(b[0].shape)
print(b[0,0].shape)
print(b[0,0])
print(b[0,1])"""