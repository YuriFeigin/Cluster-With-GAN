import tensorflow as tf
from models.ResidualBlocks import *


def z_trans(z,is_training):
    with tf.variable_scope("z_trans", reuse=tf.AUTO_REUSE) as scope:
        output = tf.layers.dense(z, 256, tf.nn.relu)
        output = tf.layers.batch_normalization(output, -1, 0.9, training=is_training)
        output = tf.layers.dense(output, 256, tf.nn.relu)
        output = tf.layers.batch_normalization(output, -1, 0.9, training=is_training)
        output = tf.layers.dense(output, 64)
    return output


def x_generator1(in_,labels,DIM,is_training,image_size, reuse):
    cur_size = image_size[0]
    while cur_size >= 8 and cur_size % 2 == 0:
        cur_size = int(cur_size/2)
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        output = tf.layers.dense(in_, cur_size*cur_size*DIM, name='Input')
        output = tf.reshape(output, [-1, cur_size, cur_size, DIM])
        i=1
        while cur_size != image_size[0]:
            output = ResidualBlock_Up('ResidualBlock_Up.'+str(i), DIM, 3,is_training, inputs=output, labels=labels)
            cur_size = output.get_shape().as_list()[1]
            i += 1
        output = NormalizeG('OutputN', output,is_training,labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output,image_size[2],3, padding='same', name='Output')
        output = tf.tanh(output)
        return output


def x_generator2(in_,labels,DIM,is_training,image_size, reuse):
    cur_size = image_size[0]
    n_layer = -1
    while cur_size >= 8 and cur_size % 2 == 0:
        cur_size = int(cur_size/2)
        n_layer += 1
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        factor = min(2**n_layer,8)
        output = in_
        output = tf.layers.dense(output, cur_size*cur_size*DIM*factor, name='Input')
        output = tf.reshape(output, [-1, cur_size, cur_size, DIM*factor])
        i=1
        while cur_size!=image_size[0]:
            factor = min(2**(n_layer-i+1), 8)
            output = ResidualBlock_Up('ResidualBlock_Up.'+str(i), DIM*factor, 3,is_training, inputs=output, labels=labels)
            cur_size = output.get_shape().as_list()[1]
            i += 1
        output = NormalizeG('OutputN', output,is_training,labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output,image_size[2],3, padding='same', name='Output')
        output = tf.tanh(output)
        return output