from PIL import Image, ImageDraw, ImageFont
import random
from typing import List
import numpy as np
import os

class img_creation:

    def __init__(self,
                fontdir = "C:/Windows/Fonts" ,
                script_path = False , 
                img_rel_path ="\\tmp_imgs\\",
                letters_ascii = range(33,126),
                fontsizes = [24,22,20,18],
                picture_size = [42,60],
                output_size = [30,30],
                position_variance = 0,
                distortion_amp = 0,
                rotation_amp = 0,
                use_edge_finder = False, 
                edge_finder_border = 0,
                contrast = 1,
                inverse_colors = False, 
                use_RGB = False # Not implemented
                ):
        if (not script_path) or (not os.path.exists(script_path)):
            script_path = os.path.dirname(os.path.realpath(__file__))
            os.chdir(script_path)
        
        if img_rel_path[:1]!="\\" and img_rel_path[:1]!="/": img_rel_path = "\\" + img_rel_path
        if img_rel_path[-1:]!="\\" and img_rel_path[-1:]!="/": img_rel_path = img_rel_path + "\\"

        self.img_path = script_path+img_rel_path
        if not os.path.exists(self.img_path): os.mkdir(self.img_path)
        self.fontdir = fontdir

        
        letters_ascii = np.asarray(letters_ascii,dtype=np.int64)
        fontsizes = np.asarray(fontsizes,dtype=np.int64)
        picture_size = np.asarray(picture_size,dtype=np.int64)

        self.letters_ascii = letters_ascii if len(letters_ascii)>0 else np.asarray(range(65,90))
        self.fontsizes = fontsizes if len(fontsizes)>0 else np.asarray([20,22])
        self.picture_size = picture_size if len(picture_size)>0 and len(picture_size)==2 else np.asarray([40,60])
        self.posvar = abs(position_variance)
        distortion_amp = abs(distortion_amp)
        self.distortion_amp = distortion_amp if distortion_amp > 1 or not distortion_amp else 1/distortion_amp
        self.rotation_amp = abs(rotation_amp)
        self.output_size = output_size
        self.use_edge_finder = False # Needs rework
        self.edge_finder_border = abs(edge_finder_border)
        self.contrast = abs(contrast)
        self.inverse_colors = inverse_colors
        self.use_RGB = use_RGB

        # Check if Values can cause problems

        if position_variance + 10 > picture_size[0] or position_variance + 15 > picture_size[1]: print("Warning: The position variance may be too big for this picturesize")
        for i in self.fontsizes:
            if i * 1.2 >  self.picture_size[0] or i * 2.0 >self.picture_size[1] :
                print("Warning: The fontsize "+str(i)+" may be too big for this picturesize")
            else:
                if i * 1.2 + position_variance >  self.picture_size[0] or i * 2.0 + position_variance >self.picture_size[1]:
                    print("Warning: The combination of fontsize "+str(i)+" and position variance may be too big for this picturesize")
            


    def get_fonts(self,
                amount=1000,
                rf= ["holomdl2.ttf","marlett.ttf","segmdl2.ttf","symbol.ttf","webdings.ttf","wingding.ttf"],
                use_test_font=False
                ):
        try:
            if amount<1:
                amount=1000000000
            fnts = os.listdir(self.fontdir)
            helper = os.listdir(self.fontdir)
            i = 0

        # Only using Fonts with in ttf-Format
        # may get extended in the Future

            for f in helper:
                if f[-3:]!="ttf":
                    fnts.remove(f)
                    i+=1
            
            for i in range(max(len(fnts)-amount,0)):
                fnts.remove(fnts[round(len(fnts)*random.random()-0.5)])    
            del helper

            for f in rf: # Remove unwanted Fonts
                if f in fnts: fnts.remove(f)

            if not hasattr(self,"test_font"): self.test_font=""
            if use_test_font: # Mark TestFont and Remove from Training Creation
                if not os.path.isfile(use_test_font): 
                    if not os.path.isfile("C:/Windows/Fonts/"+use_test_font): 
                        print("Test font not found, use Random")
                        use_test_font=fnts[round(len(fnts)*random.random()-0.5)]
                    else:
                        use_test_font= "C:/Windows/Fonts/"+one_font_mode
                self.test_font = use_test_font
            if self.test_font in fnts:
                fnts.remove(self.test_font)

            print(" - Found "+str(len(fnts))+" Fonts. Will use "+str(min(len(fnts),amount))+" Fonts -"+
                (" \n - "+self.test_font +" Will be TestFont! - " if use_test_font or self.test_font!="" else ""))

            self.fnts = fnts
            return fnts
        except:
            print("Failed")
            return None    

    def edge_finder(self,img,bcolor):
        img_data = np.array(img)
        left = len(img_data[0])
        top = len(img_data)
        right = 0
        bottom = 0
        border= self.edge_finder_border

        for i in range(len(img_data)):
            for j in range(len(img_data[0])):
                if img_data[i,j] != bcolor:
                    left = min([left,j])
                    top = min(top,i)
                    right = max([right,j])
                    bottom = max(bottom,i)

        img = img.crop(box=[left-border,top-border,right+border,bottom+border])
        return img

    def distortion_manager(self,img,bcolor,direction_x, distortion):
        max_side = max((img.width,img.height))
        longer_side = int(max_side+distortion*self.distortion_amp*max_side)
        pic_size = (longer_side,max_side) if direction_x else (max_side, longer_side) 
        img2 = Image.new('L' if isinstance(bcolor,int) else 'RGB', pic_size, color = bcolor)
        #img = Image.fromarray(img)
        img2.paste(img,box=(round(img2.width/2-img.width/2),round(img2.height/2-img.height/2)))
        ImageDraw.Draw(img2)
        return img2 #numpy.array(img2)

    def resize_to_output(self,img):
        img = img.resize([30,30])
        return img

    def rotate_img(self, img, bcolor, rotation_deg):
        img = img.rotate(rotation_deg, fillcolor=bcolor)
        return img

    def make_letter(self,letter,fnt,size,tcolor=0,bcolor=255):
        # contrast to be checked?

        fnt = ImageFont.truetype(fnt, size)

        img = Image.new('L' if isinstance(bcolor,int) else 'RGB', tuple(self.picture_size), color = bcolor)

        if len(str(letter))>1:
            if isinstance(letter,int):
                letter = chr(letter)
        if len(str(letter))>1:
            letter = str(letter)[:1]

        d = ImageDraw.Draw(img)

        posL = round(self.picture_size[0]/2-size/2+0 if self.posvar==0 else self.picture_size[0]/2-size/2+0+(random.random()-0.5)*self.posvar)
        posT = round(self.picture_size[1]/2-size/2+0 if self.posvar==0 else self.picture_size[1]/2-size/2+0+(random.random()-0.5)*self.posvar)

        d.text((posL,posT), letter , font=fnt, fill=tcolor)
       
        if self.use_edge_finder: img = self.edge_finder(img,bcolor)
        
        return img 

    def make_alphabeth(self, save_to_files=False, label_file_delimiter=";", one_font_mode=False):
        BGColor = 255
        TextColor = 0
        
        if not hasattr(self, 'fnts'): self.get_fonts()
        if one_font_mode:
            if not os.path.isfile(one_font_mode): 
                if not os.path.isfile("C:/Windows/Fonts/"+one_font_mode): 
                    print("Font not found, use Standard")
                    one_font_mode=False
                else:
                    one_font_mode= "C:/Windows/Fonts/"+one_font_mode
        alphabeth =[]
        label_file = []
        i = 0
        for fsize in self.fontsizes:
            for letter in self.letters_ascii:
                #Contrast and Inversing only Greyscale
                if not self.use_RGB:
                    if self.contrast < 1:
                        lower = int((1-self.contrast)*255*random.random())
                        TextColor=lower
                        BGColor=int(lower+self.contrast*255)
                    if self.inverse_colors and random.randint(0,1):
                        helper = BGColor
                        BGColor=TextColor
                        TextColor=helper

                l = self.make_letter(
                    chr(letter),
                    self.fnts[random.randrange(0,len(self.fnts))] if not one_font_mode else one_font_mode,
                    fsize,
                    TextColor,      # text-color
                    BGColor    # bg-color
                )
                # Distortion
                l = self.distortion_manager(
                        l,
                        BGColor,
                        random.randint(0,1),
                        (abs(random.gauss(0,0.4))/3)
                        )
                # Rotation 
                l = self.rotate_img(
                    l,
                    BGColor,
                    random.gauss(0,0.3)*self.rotation_amp
                )
                #l = self.resize_to_output(l) # Resize to tf size
                if save_to_files: 
                    label_file.append(chr(letter) + label_file_delimiter + str(i)+"\n")
                    l.save(self.img_path+str(i)+".bmp")

                # Numpy Output 
                l = np.array(l)
                alphabeth.append((l,chr(letter)))
                i+=1


        if save_to_files:
            f = open(self.img_path+"labels.txt","w")
            f.writelines(label_file)
            f.close()
        print (["i=",i,len(alphabeth)])
        
        return np.asarray(alphabeth)
        
        
        
        
        
        #if self.distortion_amp > 0: img = self.distortion_manager(img,bcolor,)


