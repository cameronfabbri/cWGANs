'''

   This is a testing file for generator.py. Here I'll be interpolating along
   the attributes and seeing if it generates anything interesting. Need to
   remember to also check the nearest neighbor in the training set just in
   case it interpolates along attributes that were trained on.

'''

import tensorflow.contrib.layers as tcl
import matplotlib.pyplot as plt
import cPickle as pickle
import tensorflow as tf
import numpy as np
import requests
import random
import math
import gzip
import os
from generator import *
#from scipy import interpolate

batch_size = 32

from tf_ops import *

def test(train_images, train_shapes, train_cont, test_images, test_shapes, test_cont):
      
   CHECKPOINT_DIR='checkpoints_generator/'

   with tf.Graph().as_default():
      global_step = tf.Variable(0, trainable=False, name='global_step')

      # the true images
      images = tf.placeholder(tf.float32, [batch_size, 64, 64, 1])

      # discrete attributes
      shape = tf.placeholder(tf.float32, [batch_size, 3]) # square, ellipse, or heart

      # continuous attribute vector
      c_attribute = tf.placeholder(tf.float32, [batch_size, 4], name='c_attribute')

      # generate image from all attributes
      gen_images = netG(shape, c_attribute)

      # saver for the model
      saver = tf.train.Saver(tf.all_variables())

      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)

      try: os.makedirs(CHECKPOINT_DIR+'interpolation_images/')
      except: pass

      ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
      if ckpt and ckpt.model_checkpoint_path:
         try:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model restored'
         except:
            print 'Could not restore model'
            pass

      step = int(sess.run(global_step))

      num_train = len(train_images)
      num_test  = len(test_images)

      idx = np.random.choice(np.arange(num_test), batch_size, replace=False)
      batch_images = test_images[idx]
      batch_shape  = test_shapes[idx]
      batch_cont   = test_cont[idx]

      # save the original image first
      gen_imgs = sess.run([gen_images],
      feed_dict={images:batch_images,
                 shape:batch_shape,
                 c_attribute:batch_cont})[0]
      for gen in gen_imgs:
         gen = np.squeeze(gen)
         plt.imsave(CHECKPOINT_DIR+'interpolation_images/original_gen.png', gen)
         break

      # [scale, orientation, x_pos, y_pos]
      
      ''' this is changing only the ORIENTATION
      scales = batch_cont[0][0]
      x_pos = batch_cont[0][-2]
      y_pos = batch_cont[0][-1]
      orien = np.linspace(0.0, 1.0, num=batch_size)
      for i in range(len(batch_shape)):
         batch_shape[i] = np.array([1,0,0]) # square
         #batch_shape[i] = np.array([0,1,0]) # ellipse
         #batch_shape[i] = np.array([0,0,1]) # heart
         #batch_shape[i] = batch_shape[0] # original image shape
         c = np.array([scales, orien[i], x_pos, y_pos])
         batch_cont[i] = c
      '''

      ''' this is changing only the SCALE
      scales = np.linspace(0.0, 1.0, num=batch_size)
      x_pos = batch_cont[0][-2]
      y_pos = batch_cont[0][-1]
      orien = batch_cont[0][1]
      for i in range(len(batch_shape)):
         batch_shape[i] = np.array([1,0,0]) # square
         #batch_shape[i] = np.array([0,1,0]) # ellipse
         #batch_shape[i] = np.array([0,0,1]) # heart
         c = np.array([scales[i], orien, x_pos, y_pos])
         batch_cont[i] = c
      '''

      #''' this is changing only the X POS
      scales = batch_cont[0][0]
      x_pos = np.linspace(0.0, 1.0, num=batch_size)
      y_pos = batch_cont[0][-1]
      orien = batch_cont[0][1]
      for i in range(len(batch_shape)):
         batch_shape[i] = np.array([1,0,0]) # square
         #batch_shape[i] = np.array([0,1,0]) # ellipse
         #batch_shape[i] = np.array([0,0,1]) # heart
         c = np.array([scales, orien, x_pos[i], y_pos])
         batch_cont[i] = c
      #'''

      gen_imgs = sess.run([gen_images],
      feed_dict={images:batch_images,
                 shape:batch_shape,
                 c_attribute:batch_cont})[0]

      c = 0
      for gen in gen_imgs:
         gen = np.squeeze(gen)
         plt.imsave(CHECKPOINT_DIR+'interpolation_images/'+str(step)+'_'+str(c)+'_gen.png', gen)
         c += 1

      exit()

# TODO save out train/test arrays
def main(argv=None):

   # if previous train/test splits are made, load them
   if os.path.isfile('data/train_images.npy'):
      print 'Loading previous splits...'
      train_images = np.load('data/train_images.npy')
      test_images  = np.load('data/test_images.npy')
      train_shapes = np.load('data/train_shapes.npy')
      test_shapes  = np.load('data/test_shapes.npy')
      train_cont   = np.load('data/train_cont.npy')
      test_cont    = np.load('data/test_cont.npy')
   else:

      # dsprites numpy array
      try: data = np.load('/mnt/data2/dsprites/dsprites_ndarray.npz')
      except: data = np.load('data/dsprites_ndarray.npz')

      # load images and attributes
      images = data['imgs']
      attributes = data['latents_values']

      total_num = len(images)
      print 'total:',total_num

      # shuffle images and attributes together
      c = list(zip(images, attributes))
      random.shuffle(c)

      # get images and attributes from shuffled c
      images[:], attributes[:] = zip(*c)

      # use 90% for training
      num_train = int(0.9*total_num)

      train_images_ = images[:num_train]
      test_images_  = images[num_train:]

      # define empty numpy arrays to load images in
      train_images = np.empty((len(train_images_), 64, 64, 1), dtype=np.float32)
      test_images  = np.empty((len(test_images_), 64, 64, 1), dtype=np.float32)

      # reshape images from (64,64) -> (64,64,1)
      i = 0
      for img in train_images_:
         img = img_norm(np.expand_dims(img, 2))
         train_images[i, ...] = img
         i += 1
      i = 0
      for img in test_images_:
         img = img_norm(np.expand_dims(img, 2))
         test_images[i, ...] = img
         i += 1

      num_test = len(test_images)
      print 'num_train:',num_train
      print 'num_test:',num_test

      # split train and test attributes
      train_attributes = attributes[:num_train]
      test_attributes  = attributes[num_train:]

      # discrete attributes are shape: square, ellipse, heart. So 3-dim one-hot vector
      train_shapes = np.empty((num_train, 3), dtype=np.float32)

      # 4 continuous attributes: scale, orientation, x_pos, y_pos
      train_cont   = np.empty((num_train, 4), dtype=np.float32)

      # same for testing
      test_shapes = np.empty((num_test, 3), dtype=np.float32)
      test_cont   = np.empty((num_test, 4), dtype=np.float32)

      i = 0
      for ta in train_attributes:
         # get the shape number and put it in range [0,1,2]
         shape_ta = int(ta[1]-1)
         # create empty one-hot vector for the shape
         s_ta = np.zeros(3)
         # fill label
         s_ta[shape_ta] = 1
         # get continuous attributes and normalize them to range [0,1]
         cont_ta = ta[2:]

         scale       = normalize(cont_ta[0], 0.5, 1)
         orientation = normalize(cont_ta[1], 0, 2*math.pi)
         x_pos       = cont_ta[2] # these are already in [0,1] range
         y_pos       = cont_ta[3]
         
         train_shapes[i, ...] = s_ta
         train_cont[i, ...]   = np.asarray([scale, orientation, x_pos, y_pos])
         i += 1

      i = 0
      for ta in test_attributes:
         # get the shape number and put it in range [0,1,2]
         shape_ta = int(ta[1]-1)
         # create empty one-hot vector for the shape
         s_ta = np.zeros(3)
         # fill label
         s_ta[shape_ta] = 1
         # get continuous attributes and normalize them to range [0,1]
         cont_ta = ta[2:]

         scale       = normalize(cont_ta[0], 0.5, 1)
         orientation = normalize(cont_ta[1], 0, 2*math.pi)
         x_pos       = cont_ta[2] # these are already in [0,1] range
         y_pos       = cont_ta[3]
         
         test_shapes[i, ...] = s_ta
         test_cont[i, ...]   = np.asarray([scale, orientation, x_pos, y_pos])
         i += 1

      # save out numpy arrays
      np.save('data/train_images.npy', train_images)
      np.save('data/test_images.npy', test_images)
      np.save('data/train_shapes.npy', train_shapes)
      np.save('data/test_shapes.npy', test_shapes)
      np.save('data/train_cont.npy', train_cont)
      np.save('data/test_cont.npy', test_cont)

   test(train_images, train_shapes, train_cont, test_images, test_shapes, test_cont)

if __name__ == '__main__':
   tf.app.run()
