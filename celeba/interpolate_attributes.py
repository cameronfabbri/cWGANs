'''

   This interpolates between two attributes. So the z value is the same, but
   the two different y values are interpolated between.

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


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--NUM',            required=False,help='Maximum images to interpolate',  type=int,default=9)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   NUM        = a.NUM

   BATCH_SIZE = NUM

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 9), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)
   D_score = netD(gen_images, y, BATCH_SIZE, 'wgan')
   
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

   c = 0
   print 'generating data...'
   batch_z = np.random.normal(0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)

   # the two z vectors to interpolate between
   two_z = np.random.normal(0, 1.0, size=[2, 100]).astype(np.float32)
   two_z[0] = two_z[1] # use same z value

   #batch_y = np.random.choice([0, 1], size=(2, 9))
   batch_y = np.zeros((2,9))

   # bald, bangs, black_hair, blond_hair, eyeglasses, heavy_makeup, male, pale_skin, smiling
   ''' male bald to not bald
   batch_y[0][-3] = 1
   batch_y[0][0] = 1
   batch_y[1][-3] = 1
   '''

   ''' female black hair to blonde hair
   batch_y[0][2] = 1
   batch_y[1][3] = 1
   '''
   
   ''' female to male
   batch_y[1][-3] = 1
   '''

   #''' male glasses and smiling to pale no glasses and not smiling
   batch_y[0][-3] = 1
   batch_y[0][4] = 1
   batch_y[0][-1] = 1
   batch_y[0][1] = 1

   batch_y[1][-2] = 1
   batch_y[1][-3] = 1
   #'''


   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   latent_y = []
   y1 = batch_y[0]
   y2 = batch_y[1]
   x1 = two_z[0]
   x2 = two_z[1]

   print 'y1:',y1
   print 'y2:',y2

   for a in alpha:
      vector = x1*(1-a) + x2*a
      latent_vectors.append(vector)
      yv = y1*(1-a) + y2*a
      latent_y.append(yv)

   latent_y = np.asarray(latent_y)
   latent_vectors = np.asarray(latent_vectors)

   gen_imgs = sess.run([gen_images], feed_dict={z:latent_vectors, y:latent_y})[0]
   canvas = 255*np.ones((80, 62*(NUM+2), 3), dtype=np.uint8)
   start_x = 10
   start_y = 10
   end_y = start_y+64

   for img in gen_imgs:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64
      canvas[start_y:end_y, start_x:end_x, :] = img
      start_x += 64+10
   misc.imsave(OUTPUT_DIR+'interp_attr.png', canvas)
