'''
   conditional gan
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
import cv2
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops
from nets import *

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--LOSS',       required=False,help='Type of GAN loss to use', type=str,default='wgan')
   parser.add_argument('--DATASET',    required=False,help='The DATASET to use',      type=str,default='mnist')
   parser.add_argument('--DATA_DIR',   required=False,help='Directory where data is', type=str,default='./')
   parser.add_argument('--EPOCHS',     required=False,help='Maximum training steps',  type=int,default=25)
   parser.add_argument('--BATCH_SIZE', required=False,help='Batch size',              type=int,default=64)
   a = parser.parse_args()

   LOSS           = a.LOSS
   DATASET        = a.DATASET
   DATA_DIR       = a.DATA_DIR
   BATCH_SIZE     = a.BATCH_SIZE
   EPOCHS      = a.EPOCHS

   CHECKPOINT_DIR = 'checkpoints/DATASET_'+DATASET+'/LOSS_'+LOSS+'/'
   IMAGES_DIR     = CHECKPOINT_DIR+'images/'
   
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10), name='y')
   fy          = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10), name='fy')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, y, BATCH_SIZE)
   errD_fake1 = 0.5*netD(gen_images, y, BATCH_SIZE, reuse=True)

   # matching aware discriminator - send real images in with fake labels and mark as fake
   #if LOSS != 'wgan': errD_fake += netD(real_images, fy, BATCH_SIZE, reuse=True)
   errD_fake2 = 0.5*netD(real_images, fy, BATCH_SIZE, reuse=True)
   errD_fake = errD_fake1 + errD_fake2

   e = 1e-12
   if LOSS == 'gan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))

   if LOSS == 'lsgan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
      errG = tf.reduce_mean(0.5*(tf.square(errD_fake - 1)))

   if LOSS == 'wgan':
      # cost functions
      errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
      errG = tf.reduce_mean(errD_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = real_images*epsilon + (1-epsilon)*gen_images
      d_hat = netD(x_hat, y, BATCH_SIZE, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS == 'wgan':
      n_critic = 5
      beta1    = 0.0
      beta2    = 0.9
      lr       = 1e-5

   if LOSS == 'lsgan':
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.001

   if LOSS == 'gan':
      '''
      n_critic = 1
      beta1    = 0.9
      beta2    = 0.999
      lr       = 2e-5
      '''
      n_critic = 1
      beta1    = 0.0
      beta2    = 0.9
      lr       = 1e-5

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess  = tf.Session()
   sess.run(init)

   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/'+'logs/', graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)

   print 'Loading data...'
   images, annots = data_ops.load_mnist(DATA_DIR)
   test_images, test_annots = data_ops.load_mnist(DATA_DIR, mode='test')

   train_len = len(annots)
   test_len  = len(test_annots)

   print 'train num:',train_len
   
   epoch_num = step/(train_len/BATCH_SIZE)

   while epoch_num < EPOCHS:
      
      epoch_num = step/(train_len/BATCH_SIZE)
      
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = annots[idx]
         batch_images = images[idx]

         # create wrong labels
         batch_fy = []
         for lab in batch_y:
            l = np.zeros((10))
            r = random.randint(0,9)
            while r == np.argmax(lab):
               r = random.randint(0,9)
            l[r] = 1
            batch_fy.append(l)
         batch_fy = np.asarray(batch_fy)
         sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, fy:batch_fy, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = annots[idx]
      batch_images = images[idx]

      # create wrong labels
      batch_fy = []
      for lab in batch_y:
         l = np.zeros((10))
         r = random.randint(0,9)
         while r == np.argmax(lab):
            r = random.randint(0,9)
         l[r] = 1
         batch_fy.append(l)
      batch_fy = np.asarray(batch_fy)

      sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, fy: batch_fy, real_images:batch_images})
      sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, fy: batch_fy, real_images:batch_images})

      # now get all losses and summary *without* performing a training step - for tensorboard and printing
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={z:batch_z, fy:batch_fy, y:batch_y, real_images:batch_images})
      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%100 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         idx          = np.random.choice(np.arange(test_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = test_annots[idx]
         batch_images = test_images[idx]
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z, y:batch_y, real_images:batch_images})[0][0]

         num = np.argmax(batch_y[0])
         plt.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', np.squeeze(gen_imgs), cmap=cm.gray)



