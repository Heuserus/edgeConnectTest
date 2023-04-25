from skimage.feature import canny
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2

img_cont = cv2.imread("datasets/hmaps_train/ASTGTMV003_N02W071_dem114.png",cv2.IMREAD_ANYDEPTH)
print(img_cont)
img_cont = Image.fromarray(img_cont)

#img_cont = img_cont.convert('L')
img_cont.save("imgconttest.png")
#enhancer = ImageEnhance.Sharpness(img_cont)
#img_cont = enhancer.enhance(-3)
#img_cont = enhancer.enhance(4)
enhancer = ImageEnhance.Contrast(img_cont)
img_cont = enhancer.enhance(0)



img_cont = asarray(img_cont)
print(img_cont)
contr = Image.fromarray(img_cont)
contr.save("withcont.png")
edge = canny(img_cont, sigma=200)
edge = Image.fromarray(edge)
edge.save("edgetest.png")