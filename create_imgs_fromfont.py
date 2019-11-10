from PIL import Image, ImageDraw, ImageFont
import random
#import numpy
from os import listdir, path

current_path = path.abspath(__file__)[:-len(path.basename(__file__))]

def get_fonts(m=1000,rf=[]):
    if m<0:
        m=1000000000
    fnts = listdir("C:/Windows/Fonts")
    helper = listdir("C:/Windows/Fonts")
    i = 0
    for f in helper:
        if f[-3:]!="ttf":
            fnts.remove(f)
            i+=1
    
    for i in range(max(len(fnts)-m,0)):
        fnts.remove(fnts[round(len(fnts)*random.random()-0.5)])
        
    del helper

    for f in rf:
        fnts.remove(f)

    print(" - Found "+str(len(fnts))+" Fonts. Will use "+str(min(len(fnts),m))+" Fonts - ")

    return fnts


def crop_imgs(img, zerocolor, border=1):
    w = img.width
    h = img.height
    helper = img.getdata()
    cropping = [0,0,w,h]

    for i in range(h):
        cut = True
        for j in range(w):
            cut = cut and helper[w*i+j]==zerocolor
        if cut and i>border:
            cropping[1]=i-border
        else:
            if i-border>0: break
    for i in range(h):
        cut = True
        for j in range(w):
            cut = cut and helper[w*(h-i)-j-1]==zerocolor
        if cut and i>border:
            cropping[3]=h-i+border
        else:
            if i-border>0: break
            
    for i in range(w):
        cut = True
        for j in range(h):
            cut = cut and helper[j*w+i]==zerocolor
        if cut and  i>border:
            cropping[0]=i-border
        else:
            if i-border>0: break
    for i in range(w):
        cut = True
        for j in range(h):
            cut = cut and helper[w*(j+1)-i-1] == zerocolor
        if cut and i>border:
            cropping[2]=w-i+border
        else:
            if i-border>0: break

    cropping = [cropping[0], cropping[1], 
        max(cropping[2],(cropping[3]-cropping[1]+cropping[0])),
        max(cropping[3],(cropping[2]-cropping[0]+cropping[1]))]

    #img = img.crop(cropping)

    #print(cropping)
    #print(img.width==img.height)
    img = img.resize([30,30],box=cropping)
    return img


def make_imgs(fnts, amount=100, fixedletter="", lettermode=0):
    labels = []
    stats = []
    if lettermode>0: print("Only lowercase is currently supported. Using casemode = 0")

    print("Creating "+str(amount)+" Pictures")

    for i in range(amount):
        fname = fnts[round(len(fnts)*random.random()-0.5)]
        fnt = ImageFont.truetype(fname, 22)
        t = chr(round(96.5+random.random()*26))
        if len(fixedletter)==1: t=fixedletter

        img = Image.new('L', (40, 60), color = 255)
        d = ImageDraw.Draw(img)
        d.text((10,10), t , font=fnt, fill=0)
        img = crop_imgs(img,255,4)

        img.save(current_path+'letterimgs\\'+str(i)+'.bmp')

        labels.append(str(t)+";"+str(i)+"\n")
        stats.append(str(fname)+"\n")
        del img
        del d

    f = open("labels.txt",'w')
    f.writelines(labels)
    f.close

    f = open("stats.txt",'w')
    f.writelines(stats)
    f.close

    return True


def create_validation_letters(fnts,text="a"):
    counter = 0
    for f in fnts:
        fnt = ImageFont.truetype(f, 22)
        img = Image.new('L', (40, 60), color = 255)

        d = ImageDraw.Draw(img)
        d.text((10,10), text , font=fnt, fill=0)
        img = crop_imgs(img,255,4)

        img.save('letterimgs\\'+f+'.bmp')
        del img
        del d
        counter += 1



#### Main call

# Fonts that dont display normal letters are excluded manually
rfnts = ["holomdl2.ttf","marlett.ttf","segmdl2.ttf","symbol.ttf","webdings.ttf","wingding.ttf"]

#fnts=["arial.ttf"]
fnts = get_fonts(m=5000,rf=rfnts)

# create_validation_letters(fnts)

if make_imgs(fnts,5000):
    print("Done")
