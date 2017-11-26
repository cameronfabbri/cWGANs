'''

   MNIST dataset generator.

   This file generates mnist the latent z vectors and the corresponding images such that the encoder can be trained

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
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
   parser.add_argument('--DATASET',        required=False,help='The DATASET to use',      type=str,default='mnist')
   parser.add_argument('--OUTPUT_DIR',     required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--MAX_GEN',        required=False,help='Maximum training steps',  type=int,default=100000)
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   DATASET        = a.DATASET
   OUTPUT_DIR     = a.OUTPUT_DIR+'/'
   MAX_GEN        = a.MAX_GEN

   BATCH_SIZE = 1

   try: os.makedirs(OUTPUT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='real_images')
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
   
   print 'Loading data...'
   images, annots = data_ops.load_mnist('./', mode='test')
   test_images, test_annots = data_ops.load_mnist('./', mode='test')

   test_len = len(test_annots)

   step = 0

   latents = []
   oimages = []

   # stores the image name and true label
   lf = open(OUTPUT_DIR+'labels.txt', 'a')

   print 'generating data...'
   #while step < MAX_GEN:
   for step in tqdm(range(MAX_GEN)):
      idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
      batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = annots[idx]
      batch_images = images[idx]
      gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})[0][0]

      num = np.argmax(batch_y[0])
      #plt.imsave(OUTPUT_DIR+'image_'+str(step)+'.png', np.squeeze(gen_imgs), cmap=cm.gray)

      latents.append(batch_z[0])
      oimages.append(gen_imgs)
      lf.write(str(num)+'\n')
      step += 1

   latents = np.asarray(latents)
   oimages = np.asarray(oimages)
   np.save(OUTPUT_DIR+'latents.npy', latents)
   np.save(OUTPUT_DIR+'images.npy', oimages)
   lf.close()
