import create_imgs_fromfont_class
import numpy as np
import os

TF_PICTURE_SIZE = [30,30]
CREATOR_PICTURE_SIZE = [42,60]
FONTSIZES = [26,24,23,22,21,20,19]
LETTER_ASCII = range(33,126)

BATCH_CREATION = 10
POS_VARS = [0,3,5,7,10]
DISTORTION_AMPS = [1.,2.,4.,1.5]
ROT_AMPS = [0,10,30,45]
CONTRASTS = [1.,0.8,0.6,0.2]
inverse_colors = True

"""BATCH_CREATION = 1
POS_VARS = [0,3]
DISTORTION_AMPS = [1.,2.]
ROT_AMPS = [0,30]
CONTRASTS = [1.]
inverse_colors = True
"""
img_rel_path = "\\imgs\\"

def make_directory(i_path,pos_var,dist_amp,rot_amp,contrast):
  new_img_path_rel = "letters_p{}_d{}_r{}_c{}_".format(pos_var,dist_amp,rot_amp,contrast)
  if os.path.exists(i_path+new_img_path_rel+"0\\"):
    i = 0
    for j in os.listdir(i_path):
      if not os.path.exists(i_path+new_img_path_rel+str(i)+"\\"): 
        new_img_path_rel = new_img_path_rel+str(i)+"\\"
        break
      else:
        i+=1
  else:
    new_img_path_rel = new_img_path_rel+"0\\"
  os.mkdir(i_path+new_img_path_rel)
  return(img_rel_path+new_img_path_rel)


script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
img_path = script_path+img_rel_path
if not os.path.exists(img_path): os.mkdir(img_path)

image_creator = create_imgs_fromfont_class.img_creation()


for batch_counter in range(BATCH_CREATION):
  for pos_var in POS_VARS:
    for dist_amp in DISTORTION_AMPS:
      for rot_amp in ROT_AMPS:
        for contrast in CONTRASTS:
          image_creator.__init__(
                        position_variance=pos_var,
                        distortion_amp=dist_amp,
                        use_edge_finder=False, #Looses info for training
                        rotation_amp=rot_amp,
                        inverse_colors=True,
                        contrast=contrast,
                        letters_ascii=LETTER_ASCII,
                        fontsizes=FONTSIZES,
                        picture_size=CREATOR_PICTURE_SIZE,
                        output_size=TF_PICTURE_SIZE,
                        img_rel_path= make_directory(img_path,pos_var,dist_amp,rot_amp,contrast)
                        )
          image_creator.make_alphabeth(save_to_files=True)
          

