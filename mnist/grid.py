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


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='checkpoint directory',type=str)
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--NUM',            required=False,help='Maximum images to interpolate',  type=int,default=9)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   NUM            = a.NUM

   BATCH_SIZE = NUM

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10), name='y')

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
   
   step = 0

   c = 0
   print 'generating data...'
   batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)

   # the four z vectors to interpolate between
   f_z = np.random.normal(-1.0, 1.0, size=[4, 100]).astype(np.float32)

   y1 = np.zeros((10))
   y2 = np.zeros((10))
   y3 = np.zeros((10))
   y4 = np.zeros((10))
   
   y1[1] = 1
   y2[2] = 1
   y3[3] = 1
   y4[4] = 1
   
   # four corners
   z1 = f_z[0]
   z2 = f_z[1]
   z3 = f_z[1]
   z4 = f_z[1]

   # first generate the first column - z1 to z3
   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   latent_y = []

   for a in alpha:
      vector = z1*(1-a) + z3*a
      latent_vectors.append(vector)
      vy = y1*(1-a) + y3*a
      latent_y.append(vy)
   latent_y = np.asarray(latent_y)
   latent_vectors = np.asarray(latent_vectors)
   
   gen_imgs1 = sess.run([gen_images], feed_dict={z:latent_vectors, y:latent_y})[0]

   # generate last column - z2 to z4
   latent_vectors = []
   latent_y = []

   for a in alpha:
      vector = z2*(1-a) + z4*a
      latent_vectors.append(vector)
      vy = y2*(1-a) + y4*a
      latent_y.append(vy)
   latent_y = np.asarray(latent_y)
   latent_vectors = np.asarray(latent_vectors)

   gen_imgs2 = sess.run([gen_images], feed_dict={z:latent_vectors, y:latent_y})[0]

   canvas = 255*np.ones((38, 300), dtype=np.uint8)
   start_x = 5
   start_y = 5
   end_y = start_y+28

   for img in gen_imgs1:
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+28
      img = np.reshape(img, [28, 28])
      canvas[start_y:end_y, start_x:end_x] = img
      start_x += 28+5
   plt.imsave(OUTPUT_DIR+'interpolate.png', np.squeeze(canvas), cmap=cm.gray)
