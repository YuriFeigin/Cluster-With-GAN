import tensorflow as tf
from models.ResidualBlocks import *
import models.z_generators as z_generators
import models.x_generators as x_generators

def z_generator(in_,z_len,DIM, is_training,image_size, reuse):
    with tf.variable_scope("Encoder", reuse=reuse) as scope:
        return z_generators.z_generator1(in_,[None,None],z_len,DIM,True, is_training,image_size, reuse)

def x_generator(in_,DIM,is_training,image_size, reuse):
    with tf.variable_scope("Decoder", reuse=reuse) as scope:
        return x_generators.x_generator1(in_,[None,None],DIM,is_training,image_size, reuse)

def discriminator(in_x, in_z, DIM, is_training, image_size, reuse):
    with tf.variable_scope("Discriminator", reuse=reuse) as scope:
        out_x = z_generators.z_generator12(in_x,None,None,DIM,True, is_training,image_size, reuse)
        out_x = tf.layers.dense(out_x, 512)
        
        out_z = in_z
        out_z = tf.layers.dropout(out_z, 0.2, training=is_training)
        out_z = tf.layers.dense(out_z, 512)
        out_z = nonlinearity(out_z)
        out_z = tf.layers.dropout(out_z, 0.5, training=is_training)
        out_z = tf.layers.dense(out_z, 512)
        out_z = nonlinearity(out_z)

        out_c = tf.concat([out_x, out_z], axis=-1)
        out_c = tf.layers.dropout(out_c, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1024)
        out_c = nonlinearity(out_c)
        out_c = tf.layers.dropout(out_c, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1024)
        out_c = nonlinearity(out_c)
        out_c = tf.layers.dropout(out_c, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1)
        return out_c