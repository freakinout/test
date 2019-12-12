import tensorflow as tf
import numpy as np
import os
import random
import create_imgs_fromfont_class

use_font = "arial.ttf"
verbose = False
letter_range = range(65,91)


script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
models_path = script_path + "\\mymodels\\letter_recognition\\"


img_creator = create_imgs_fromfont_class.img_creation(
    position_variance=0,
    distortion_amp=0,
    use_edge_finder=False, #Looses info for training
    edge_finder_border=0,
    rotation_amp=0,
    inverse_colors=False,
    contrast=1
    )
  
c = []
fnts =  img_creator.get_fonts()

for i in letter_range:
    fnt = fnts[round(random.randrange(0,len(fnts)))] if use_font=="" else use_font
    b = img_creator.make_letter(chr(i),fnt,15)
    b = img_creator.resize_to_output(b)
    #b.save(img_path+"test_"+str(i)+".bmp")
    b = np.array(b)
    b = tf.image.convert_image_dtype(b,tf.float32)
    b = tf.reshape(b,(30,30,1))
    c.append(b)

c = np.asarray(c)


models = os.listdir(models_path)
print(models)

models.remove("logs")
models.remove("index_list.txt")

model_name = models[len(models)-2]

print("use model: {}".format(model_name))
model = tf.keras.models.load_model(models_path+model_name)

prediction = model.predict(x=c)


if verbose:
  print("predition of first element:")
  print(prediction[0])


f = open(models_path+model_name+"\\index_list.txt")
lines = f.readlines()
f.close()

correct_counter = 0
j = 0
for p in prediction:
    l = np.argmax(p)
    if verbose : print(l)
    if str.strip(lines[l]) == chr(j+min(letter_range)): correct_counter += 1
    j += 1



print("Correct read: {} of {}".format(correct_counter,j))