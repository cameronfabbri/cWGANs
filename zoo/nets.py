import tensorflow as tf
import tensorflow.contrib.layers as tcl
from tf_ops import *

'''
   Generator network
   batch norm before activation function
'''
def netG(z, y, BATCH_SIZE):

   # concat attribute y onto z
   z = tf.concat([z,y], axis=1)
   z = tcl.fully_connected(z, 4*4*512, activation_fn=tf.identity, scope='g_z')
   #z = tcl.batch_norm(z)
   z = tf.reshape(z, [BATCH_SIZE, 4, 4, 512])
   #z = tf.nn.relu(z)

   conv1 = tcl.convolution2d_transpose(z, 512, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv1')
   
   conv2 = tcl.convolution2d_transpose(conv1, 256, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv2')
   
   conv3 = tcl.convolution2d_transpose(conv2, 128, 5, 2, normalizer_fn=tcl.batch_norm, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv3')
   
   conv4 = tcl.convolution2d_transpose(conv3, 3, 5, 2, activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='g_conv4')

   print 'z:',z
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print
   print 'END G'
   print
   tf.add_to_collection('vars', z)
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   return conv4

'''
   Discriminator network. No batch norm when using WGAN
'''
def netD(input_images, y, BATCH_SIZE, LOSS, reuse=False):

   print 'DISCRIMINATOR reuse = '+str(reuse)
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      y_dim = int(y.get_shape().as_list()[-1])

      # reshape so it's batchx1x1xy_size
      y = tf.reshape(y, shape=[BATCH_SIZE, 1, 1, y_dim])
      input_ = conv_cond_concat(input_images, y)

      conv1 = tcl.conv2d(input_, 64, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv1')
      conv1 = lrelu(conv1)
      
      conv2 = tcl.conv2d(conv1, 128, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv2')
      if LOSS != 'wgan': conv2 = tcl.batch_norm(conv2)
      conv2 = lrelu(conv2)

      conv3 = tcl.conv2d(conv2, 256, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv3')
      if LOSS != 'wgan': conv3 = tcl.batch_norm(conv3)
      conv3 = lrelu(conv3)

      conv4 = tcl.conv2d(conv3, 512, 5, 2, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv4')
      if LOSS != 'wgan': conv4 = tcl.batch_norm(conv4)
      conv4 = lrelu(conv4)

      conv5 = tcl.conv2d(conv4, 1, 4, 1, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='d_conv5')

      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print 'END D\n'

      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', conv5)

      return conv5

def activate(x, ACTIVATION):
   if ACTIVATION == 'lrelu': return lrelu(x)
   if ACTIVATION == 'relu':  return relu(x)
   if ACTIVATION == 'elu':   return elu(x)
   if ACTIVATION == 'swish': return swish(x)

'''
   Encoder
'''
def encZ(x, ACTIVATION):

   conv1 = tcl.conv2d(x, 32, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv1')
   conv1 = activate(conv1, ACTIVATION)
   
   conv2 = tcl.conv2d(conv1, 64, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv2')
   conv2 = activate(conv2, ACTIVATION)

   conv3 = tcl.conv2d(conv2, 128, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv3')
   conv3 = activate(conv3, ACTIVATION)

   conv4 = tcl.conv2d(conv3, 256, 5, 2, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='conv4')
   conv4 = activate(conv4, ACTIVATION)

   conv4_flat = tcl.flatten(conv4)

   fc1 = tcl.fully_connected(conv4_flat, 4096, activation_fn=tf.identity, normalizer_fn=tcl.batch_norm, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc1')
   fc1 = activate(fc1, ACTIVATION)
   fc1 = tcl.dropout(fc1, 0.5)
   
   fc2 = tcl.fully_connected(fc1, 100, activation_fn=tf.identity, weights_initializer=tf.random_normal_initializer(stddev=0.02), scope='fc2')
   
   # relu to match the [0,1] range from the distribution
   fc2 = relu(fc2)

   print 'input:',x
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'fc1:',fc1
   print 'fc2:',fc2
   print 'END ENCODER\n'
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', fc1)
   tf.add_to_collection('vars', fc2)

   return fc2

