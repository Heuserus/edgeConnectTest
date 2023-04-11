from skimage.feature import canny
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
#from scipy.misc import imread
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb

img_cont = Image.open("datasets/hmaps_train_contrast/ASTGTMV003_N02W072_dem4.png")
img_cont = img_cont.convert('L')
enhancer = ImageEnhance.Contrast(img_cont)
img_cont = enhancer.enhance(20)
enhancer = ImageEnhance.Brightness(img_cont)
img_cont = enhancer.enhance(1)



img_cont = asarray(img_cont)
print(img_cont)
imglook = Image.fromarray(img_cont)
imglook.save("imgconttest.png")
edge = canny(img_cont, sigma=2)
edge = Image.fromarray(edge)
edge.save("edgetest.png")