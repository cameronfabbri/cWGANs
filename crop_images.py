import os
import ntpath
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.misc as misc

images = glob('img_align_celeba/*.jpg')

def crop_center(img,cropx=64,cropy=64):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


x1 = 30
y1 = 40
for i in tqdm(images):

   filename = ntpath.basename(i)
   img = misc.imread(i)
   img = img[y1:y1+138, x1:x1+138,:]
   img = misc.imresize(img, (64,64))
   misc.imsave('img_align_celeba_cropped/'+filename, img)
