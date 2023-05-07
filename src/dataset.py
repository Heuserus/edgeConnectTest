import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageOps, ImageEnhance
#from scipy.misc import imread
from numpy import asarray
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import cv2
from skimage import measure

def contouredge(img):
    contour_image = np.zeros_like(img)

    contour_image = np.zeros_like(img,dtype=np.uint8)
    for contour_interval in range(12,60,1):
        contours = measure.find_contours(img, contour_interval**2)
        for contour in contours:
            contour = np.around(contour).astype(np.int32)
            contour_image[contour[:, 0], contour[:, 1]] = 255
    Image.fromarray(contour_image).save("midcontour.png")
    return contour_image


    


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, edge_flist, mask_flist, contrast_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)
        self.contrast_data = self.load_flist(contrast_flist)
        print(len(self.contrast_data))
        

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 2:
            self.mask = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        size = self.input_size

        # load image
        img  = cv2.imread(self.data[index], cv2.IMREAD_ANYDEPTH)

    
        img = asarray(img)
        img_cont = np.asarray(Image.open(self.contrast_data[index]))

        # create grayscale image
        img_gray = img_cont

        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        

        # load mask
        mask = self.load_mask(img, index)

        # load edge
        edge = self.load_edge(img, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

            

        return self.to16b_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma
        

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=mask).astype(float)

        if self.edge == 2:
            
            return asarray(contouredge(img))
        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = Image.open(self.edge_data[index])
            edge = asarray(edge)
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=mask)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            #TODO: Make Correct Masks
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2 , 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = Image.open(self.mask_data[mask_index])
            mask = asarray(mask)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = Image.open(self.mask_data[index])
            mask = asarray(mask)
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

        if mask_type == 7:
            #TODO: Make Correct Masks
            if random.random() < 0.5: #lines
                if random.random() < 0.5: #vertical
                    if random.random() < 0.8: #sides
                        return create_mask(imgw, imgh, imgw // 4, imgh, 0 if random.random() < 0.5 else (imgw // 4)*3 , 0)
                    else: #middles
                        return create_mask(imgw, imgh, imgw // 4, imgh, imgw//4 if random.random() < 0.5 else imgw // 2, 0)
                else:
                    if random.random() < 0.8: #sides
                        return create_mask(imgw, imgh, imgw, imgh // 4, 0, 0 if random.random() < 0.5 else (imgh // 4)*3 )
                    else: #middles
                        return create_mask(imgw, imgh, imgw, imgh // 4, 0, imgh//4 if random.random() < 0.5 else imgh // 2)  
            else: #blocks
                if random.random() < 0.8:
                         return create_mask(imgw, imgh, imgw // 2, imgh // 2, 0 if random.random() < 0.5 else imgw // 2 , 0 if random.random() < 0.5 else imgh // 2)
                else:
                    if random.random() < 0.5:
                        if random.random() < 0.5:
                            return create_mask(imgw, imgh, imgw // 2, imgh // 2, imgw//4 , 0 if random.random() < 0.5 else imgh // 2)
                        else:
                            return create_mask(imgw, imgh, imgw // 2, imgh // 2, 0 if random.random() < 0.5 else imgw // 2 , imgh // 4)
                    else:
                        return create_mask(imgw, imgh, imgw // 2, imgh // 2, imgw//4 , imgw//4)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        
        img_t = F.to_tensor(img).float()

               
        return img_t

    def to16b_tensor(self,img):
        img = Image.fromarray(img)
        
        img_t = F.to_tensor(img).float()

                
        return img_t / 65536.0

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        #img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize((height,width)))

        return img

    def load_flist(self, flist):
        
        if isinstance(flist, list):
            return flist
        
        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            
            if os.path.isdir(flist):
                
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    

        
