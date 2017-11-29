'''

   This interpolates between two generated images

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
   parser.add_argument('--NUM',            required=False,help='Maximum images to interpolate',  type=int,default=5)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   OUTPUT_DIR     = a.OUTPUT_DIR
   NUM        = a.NUM

   BATCH_SIZE = NUM

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
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
   
   #$print 'Loading data...'
   #images, annots, test_images, test_annots = data_ops.load_celeba(DATA_DIR)

   #test_images = images
   #test_annots = annots

   #test_len = len(test_annots)

   step = 0

   info_dict = {}

   c = 0
   print 'generating data...'
   batch_z = np.random.normal(0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)

   # the two z vectors to interpolate between
   two_z = np.random.normal(0, 1.0, size=[2, 100]).astype(np.float32)

   batch_y = np.random.choice([0, 1], size=(9))
   batch_y[-3] = 1
   batch_y = np.asarray([batch_y,]*BATCH_SIZE) # repeat same attribute for all images
   print batch_y[0]
   # put this in if not using any attributes
   #batch_y = np.zeros((BATCH_SIZE,9))

   alpha = np.linspace(0,1, num=NUM)
   latent_vectors = []
   x1 = two_z[0]
   x2 = two_z[1]

   for a in alpha:
      vector = x1*(1-a) + x2*a
      latent_vectors.append(vector)

   latent_vectors = np.asarray(latent_vectors)
   #print latent_vectors

   gen_imgs = sess.run([gen_images], feed_dict={z:latent_vectors, y:batch_y})[0]
   i = 0
   canvas = 255*np.ones((80, 64*(NUM+2), 3), dtype=np.uint8)

   start_x = 10
   start_y = 10
   end_y = start_y+64

   for img in gen_imgs:
      
      img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
      img *= 255.0/img.max()
      end_x = start_x+64

      canvas[start_y:end_y, start_x:end_x, :] = img
      
      start_x += 64+10

   misc.imsave(OUTPUT_DIR+'canvas.png', canvas)
