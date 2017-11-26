'''

   Getting the scores for images generated.

   Generate a batch of images, get their scores from D, reject those with a score
   below a threshold. This should give us very good generated images for the training set.

'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops
from nets import *

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE',     required=False,help='Batch size',          type=int,default=64)
   parser.add_argument('--CHECKPOINT_DIR', required=True,help='Checkpoint directory', type=str,default='./')
   parser.add_argument('--DATA_DIR',       required=True,help='Data directory', type=str,default='./')
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   DATA_DIR       = a.DATA_DIR
   IMAGES_DIR     = CHECKPOINT_DIR+'test_images/'
   LOSS           = 'wgan'

   try: os.makedirs(IMAGES_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 9), name='y')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)
   D_score = netD(gen_images, y, BATCH_SIZE, 'wgan')
   #D_score = tf.reduce_mean(D_score)

   saver   = tf.train.Saver(max_to_keep=1)
   init    = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess    = tf.Session()
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   print 'Loading data...'
   images, annots, test_images, test_annots = data_ops.load_celeba(DATA_DIR)

   test_images = images
   test_annots = annots

   train_len = len(annots)
   test_len  = len(test_annots)

   idx     = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
   batch_z = np.random.normal(0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
   batch_y = test_annots[idx]

   gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]

   # get score from the discriminator
   dscores = sess.run([D_score], feed_dict={z:batch_z, y:batch_y})[0]

   c = 1
   f = open(IMAGES_DIR+'/scores.txt', 'w')
   f.close()
   f = open(IMAGES_DIR+'/scores.txt', 'a')
   for im,score in zip(gen_imgs,dscores):
      image_name = IMAGES_DIR+'img_'+str(c)+'.png'
      misc.imsave(image_name, im)
      f.write(str(np.mean(score))+'\n')
      c += 1
   f.close()
