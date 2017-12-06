'''
   This file generates celeba latent z vectors and the corresponding images such that the encoder can be trained
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
   parser.add_argument('--DATASET',        required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--DATA_DIR',       required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory to save data', type=str,default='./')
   parser.add_argument('--MAX_GEN',        required=False,help='Maximum training steps',  type=int,default=100000)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   DATASET        = a.DATASET
   OUTPUT_DIR     = a.OUTPUT_DIR
   MAX_GEN        = a.MAX_GEN
   DATA_DIR       = a.DATA_DIR

   BATCH_SIZE = 64

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 14), name='y')

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
   
   print 'Loading data...'
   images, annots, test_images, test_annots, _ = data_ops.load_galaxy(DATA_DIR)

   test_images = images
   test_annots = annots

   test_len = len(test_annots)

   step = 0

   '''
      Save the image to a folder
      write to a pickle file {image_name:label}
   '''
   info_dict = {}

   c = 0
   r = 0
   print 'generating data...'
   #for step in tqdm(range(int(MAX_GEN/BATCH_SIZE))):
   while c < MAX_GEN:
      idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
      batch_z = np.random.normal(0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y = test_annots[idx]

      gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]
      # get score from the discriminator
      dscores = sess.run([D_score], feed_dict={z:batch_z, y:batch_y})[0]

      for im,y_,z_,score in zip(gen_imgs,batch_y,batch_z,dscores):
         s = np.mean(score)
         if s > -4.0:
            image_name = OUTPUT_DIR+'img_'+str(c)+'.png'
            info_dict[image_name] = [y_, z_]
            misc.imsave(image_name, im)
            c += 1
         #else:
         #   misc.imsave('rejects/reject_'+str(r)+'.png', im)
         #   r += 1

   # write out dictionary to pickle file
   p = open(OUTPUT_DIR+'data.pkl', 'wb')
   data = pickle.dumps(info_dict)
   p.write(data)
   p.close()
   exit()
