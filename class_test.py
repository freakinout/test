import create_imgs_fromfont_class
import numpy as np

a = create_imgs_fromfont_class.img_creation(
    position_variance=0,
    distortion_amp=4,
    use_edge_finder=False, #Looses info for training
    edge_finder_border=3,
    rotation_amp=30,
    inverse_colors=True,
    contrast=0.3
    )

#b = a.make_letter("Q","C:/Windows/Fonts/arial.ttf",15,filename="mytry.bmp")
#b = np.array(b)
a.test_font = "arial.ttf"
b = a.make_alphabeth(True,one_font_mode="arial.ttf")
#print(a.fnts)

print(b.shape)
"""
print(b[0].shape)
print(b[0,0].shape)
print(b[0,0])
print(b[0,1])"""