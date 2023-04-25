import os
import random
import numpy as np
import png
import numba.cuda as cuda

from PIL import Image, ImageOps
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2

@cuda.jit
def autocontrast_kernel(img):
    sizetest = 8000
    smalltest = 0  
    h, w = cuda.grid(2)
    
    if h < img.shape[0] and w < img.shape[1]:
        img[h][w] = round(((65535 - 0) / (sizetest - smalltest)) * (img[h][w] - smalltest)/256)
        if img[h][w] > 255:
            img[h][w] = 255
        if img[h][w] < 0:
            img[h][w] = 0

def autocontrast(img):
    img_gpu = cuda.to_device(img)
    threads_per_block = (16, 16)
    blocks_per_grid_x = np.ceil(img.shape[0] / threads_per_block[0]).astype(int)
    blocks_per_grid_y = np.ceil(img.shape[1] / threads_per_block[1]).astype(int)
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    autocontrast_kernel[blocks_per_grid, threads_per_block](img_gpu)
    cuda.synchronize()

    return img_gpu.copy_to_host()

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


