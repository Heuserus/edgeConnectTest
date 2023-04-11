import os
import random
import numpy as np
import png

from PIL import Image, ImageOps
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2


def autocontrast(img):
        sizetest = 0
        smalltest = 65535   
        for h in range(len(img)):
            for w in range(len(img[h])):
                if img[h][w] > sizetest:
                    sizetest = img[h][w]
                if img[h][w] < smalltest:
                    smalltest = img[h][w]
        for h in range(len(img)):
            for w in range(len(img[h])):
                img[h][w] = round(((65535 - 0) / (sizetest - smalltest)) * (img[h][w] - smalltest)/256)
                if img[h][w] > 255:
                    img[h][w] = 255
                if img[h][w] < 0:
                    img[h][w] = 0
        return img

directory = './datasets/hmaps_val/'
directory_target = './datasets/hmaps_val_contrast/'
lst = os.listdir(directory)
count = 0
for filename in lst:
    count += 1
    print(count)
    img = cv2.imread(directory + filename,cv2.IMREAD_UNCHANGED)
    array = asarray(img)
    array.setflags(write=True)
    array = autocontrast(array)
    
    Image.fromarray(array.astype(np.uint8)).save(directory_target+filename)


