import tensorflow as tf
from models.ResidualBlocks import *

def z_generator1(in_,labels,DIM,Normalization, is_training,image_size, reuse):
    with tf.variable_scope("z_generator1", reuse=reuse) as scope:
        output = tf.layers.conv2d(in_, DIM,3, padding='same', name='Input')
        output = nonlinearity(output)
        cur_size = image_size[0]
        i=1
        while cur_size >= 8 and cur_size % 2 == 0:
            output = ResidualBlock_Down('ResidualBlock_Down.'+str(i), DIM, 3,is_training,Normalization, inputs=output)
            cur_size = output.get_shape().as_list()[1]
            i+=1
        output = NormalizeG('OutputN', output,is_training,labels)
        output = tf.layers.flatten(output)

        return output

def z_generator2(in_,labels,DIM,Normalization, is_training,image_size, reuse):
    with tf.variable_scope("z_generator2", reuse=reuse) as scope:
        output = tf.layers.conv2d(in_, DIM,3, padding='same', name='Input')
        cur_size = image_size[0]
        i=0
        while cur_size >= 8 and cur_size % 2 == 0:
            factor = min(2 ** i, 8)
            output = ResidualBlock_Down('ResidualBlock_Down.'+str(i), DIM*factor, 3,is_training,Normalization, inputs=output)
            cur_size = output.get_shape().as_list()[1]
            i += 1
        output = NormalizeG('OutputN', output,is_training,labels)
        output = tf.layers.flatten(output)
        return output

def z_generator12(in_x,labels,DIM,Normalization, is_training,image_size, reuse):
    with tf.variable_scope("z_generator12", reuse=reuse) as scope:
        output = OptimizedResBlockDisc1(in_x,DIM)
        cur_size = output.get_shape().as_list()[1]
        i = 1
        while cur_size > 8 and cur_size % 2 == 0:
            output = ResidualBlock_Down('ResidualBlock_Down.'+str(i), DIM, 3,is_training,Normalization, output, labels=labels)
            output = tf.nn.dropout(output, keep_prob=0.8)  # dropout after activator
            cur_size = output.get_shape().as_list()[1]
            i += 1
        output = ResidualBlock('ResidualBlock.1', DIM, 3, output, labels=labels)
        output = tf.nn.dropout(output, keep_prob=0.5)     #dropout after activator
        output = ResidualBlock('ResidualBlock.2', DIM, 3, output, labels=labels)
        output = tf.nn.dropout(output, keep_prob=0.5)     #dropout after activator
        output = nonlinearity(output)
        output = tf.layers.flatten(output)
        return output
