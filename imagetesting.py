import os
import random
import numpy as np

from PIL import Image, ImageOps
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2


img = cv2.imread("datasets/hmaps_train/ASTGTMV003_N02W071_dem3.png",cv2.IMREAD_UNCHANGED)

cv2.imwrite("read_test.png", img)

img_cont = ImageOps.autocontrast(img)
cv2.imwrite("cont.png", img)