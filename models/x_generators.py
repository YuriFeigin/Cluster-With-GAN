import tensorflow as tf
from models.ResidualBlocks import *


def x_generator1(in_,labels,DIM,is_training,image_size, reuse):
    cur_size = image_size[0]
    while cur_size >= 8 and cur_size % 2 == 0:
        cur_size = int(cur_size/2)
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        output = tf.layers.dense(in_, cur_size*cur_size*DIM, name='Input')
        output = tf.reshape(output, [-1, cur_size, cur_size, DIM])
        i=1
        while cur_size!=image_size[0]:
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
        n_layer+=1
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        factor = min(2**n_layer,8)
        output = tf.layers.dense(in_, cur_size*cur_size*DIM*factor, name='Input')
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