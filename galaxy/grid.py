'''

   This creates a grid of mnist images by interpolating between the four corners

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import cPickle as pickle
from tqdm import tqdm
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import cv2
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops
from nets import *

def getCanvas(canvas, gen_imgs, start_x, start_y):
   end_y = start_y+64
   for img in gen_imgs:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64
      img = np.reshape(img, [64, 64, 3])
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+5
   return canvas

def interp(x1, x2, y1, y2):
   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   latent_y = []
   for a in alpha:
      vector = x1*(1-a) + x2*a
      latent_vectors.append(vector)
      vy = y1*(1-a) + y2*a
      latent_y.append(vy)
   latent_y = np.asarray(latent_y)
   latent_vectors = np.asarray(latent_vectors)
   return latent_vectors, latent_y

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--DATA_DIR',     required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--NUM',            required=False,help='Maximum images to interpolate',  type=int,default=9)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   DATA_DIR       = a.DATA_DIR
   NUM            = a.NUM

   BATCH_SIZE = NUM

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 4), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)
   
   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise
         exit()
   
   print 'Loading data...'
   images, annots, test_images, test_annots, _ = data_ops.load_galaxy(DATA_DIR)
   test_len = len(test_annots)

   print 'generating data...'
   idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
   batch_y = test_annots[idx]
   # the four z vectors to interpolate between
   f_z = np.random.normal(-1.0, 1.0, size=[4, 100]).astype(np.float32)

   # contains rows of images
   gen_imgs = []

   y1 = batch_y[0]
   y2 = batch_y[1]
   y3 = batch_y[2]
   y4 = batch_y[3]

   # four corners
   z1 = f_z[0]
   z2 = f_z[1]
   z3 = f_z[1]
   z4 = f_z[1]

   # column 1
   latent_vectors1, latent_y1 = interp(z1, z3, y1, y3)
   gen_imgs1 = sess.run([gen_images], feed_dict={z:latent_vectors1, y:latent_y1})[0]
   #gen_imgs.append(gen_imgs1)
   
   # column 2
   latent_vectors2, latent_y2 = interp(z2, z4, y2, y4)
   gen_imgs2 = sess.run([gen_images], feed_dict={z:latent_vectors2, y:latent_y2})[0]
   #gen_imgs.append(gen_imgs2)

   # now interpolate between each row from each column
   for i in range(NUM):
      # left column z
      _z1 = latent_vectors1[i]
      # right column z
      _z2 = latent_vectors2[i]

      _y1 = latent_y1[i]
      _y2 = latent_y2[i]
      
      latent_vectors, latent_y = interp(_z1, _z2, _y1, _y2)
      gimgs = sess.run([gen_images], feed_dict={z:latent_vectors, y:latent_y})[0]
      gen_imgs.append(gimgs)

   canvas = 255*np.ones(( (NUM*5)+(NUM*64)+4, (NUM*5)+(NUM*64)+4, 3 ), dtype=np.uint8)
   start_x = 5
   start_y = 5
   for img in gen_imgs:
      canvas = getCanvas(canvas, img, start_x, start_y)
      start_y += 64+5 # new row


   misc.imsave(OUTPUT_DIR+'grid.png', np.squeeze(canvas))
