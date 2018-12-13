from PIL import Image
import sys 
import os
import subprocess

def get_imlist(path, fmt_input):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(fmt_input)]

def get_file_list(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def resizePhoto(infile,size):
    photo = Image.open(infile)
    photo =  photo.convert('RGB')
    (x,y) = photo.size
    
    if x>y:
        photo = photo.crop(((x-y)/2,0,(x+y)/2,y))
        
    else:
        photo = photo.crop((0,(y-x)/2,x,(x+y)/2))
    
        #print x_s,y_s
    out = photo.resize((size,size),Image.ANTIALIAS)
    out.save(infile)
    #print(out.size)

def numbers_to_strings(argument):
    switcher = {
        "jpg": "JPG",
        "png": "PNG",
        "pgm": "PGM",
    }
    return switcher.get(argument, "nothing")


def main(path,fmt,size):
    im_list = get_imlist(path, fmt) + get_imlist(path, numbers_to_strings(fmt))
    #print (im_list)
    for img_path in im_list:
        resizePhoto(img_path,size)

def entrance(): 
    path = "img\\"
    fmt = "jpg"
    size = 100
    file_list = get_file_list(path)
    for dir in file_list:
        main(dir,fmt,size)
    print("All images have been adjusted to the size (" + str(size) + '*' + str(size) +')')

entrance()