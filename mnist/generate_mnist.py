'''

   Generates MNIST images for testing different y values

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
   a = parser.parse_args()

   CHECKPOINT_DIR = a.CHECKPOINT_DIR
   DATASET        = a.DATASET
   OUTPUT_DIR     = a.OUTPUT_DIR+'/'

   BATCH_SIZE = 64

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
   #images, annots = data_ops.load_mnist('./', mode='test')
   test_images, test_annots = data_ops.load_mnist('./', mode='test')

   test_len = len(test_annots)
   
   n = 10 # cols
   m = 3  # rows
   num_images = n*m
   img_size = (28, 28)
   #canvas = 1*np.ones((m*img_size[0]+(10*m)+10, n*img_size[1]+(10*n)+10), dtype=np.uint8)
   canvas = 255*np.ones((200, 390), dtype=np.uint8)
   print canvas.shape

   start_x = 10
   start_y = 10
   x_ = 0
   y_ = 0

   for j in range(5):

      batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,10))
      batch_y[0][:] = 0

      # for the current style, go through all 10 labels
      for i in range(10):
         batch_y[0][i] = 1
         print 'generating', np.argmax(batch_y[0])

         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y})[0]
      
         for img in gen_imgs:
            img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
            img *= 255.0/img.max()
            end_x = start_x+28
            end_y = start_y+28
            img = np.reshape(img, [28, 28])
            canvas[start_y:end_y, start_x:end_x] = img
            start_x += 28+10
            break
   
         #plt.imsave(OUTPUT_DIR+'results.png', np.squeeze(canvas), cmap=cm.gray)

         batch_y = np.random.choice([0, 1], size=(BATCH_SIZE,10))
         batch_y[0][:] = 0

      print
      start_x = 10
      start_y = end_y + 10
   
   plt.imsave(OUTPUT_DIR+'results.png', np.squeeze(canvas), cmap=cm.gray)
   
   exit()

   num = np.argmax(batch_y[0])
   #plt.imsave(OUTPUT_DIR+'image_'+str(step)+'.png', np.squeeze(gen_imgs), cmap=cm.gray)

   latents.append(batch_z[0])
   oimages.append(gen_imgs)
   lf.write(str(num)+'\n')
