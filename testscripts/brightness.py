import os
from PIL import Image, ImageEnhance




directory = './datasets/hmaps_val_contrast/'
directory_target = './datasets/hmaps_val_highcontrast/'
lst = os.listdir(directory)
count = 0
for filename in lst:
    count += 1
    print(count)
    img = Image.open(directory + filename)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(20)
    img.save(directory_target+filename)