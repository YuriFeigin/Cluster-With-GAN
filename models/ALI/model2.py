import tensorflow as tf
from models.ResidualBlocks import *
import models.z_generators as z_generators
import models.x_generators as x_generators
import layers.gaussian_rank1

def z_generator(in_,z_len,DIM, is_training,image_size, reuse):
    with tf.variable_scope("Encoder", reuse=reuse) as scope:
        output1 = z_generators.z_generator2(in_, [None, None], DIM, True, is_training, image_size, reuse)
        output2 = tf.layers.dense(output1,z_len)
        return output2

def x_generator(in_,DIM,is_training,image_size, reuse):
    with tf.variable_scope("Decoder", reuse=reuse) as scope:
        return x_generators.x_generator2(in_,[None,None],DIM,is_training,image_size, reuse)

def discriminator(in_x, in_z,DIM, is_training,image_size, reuse):
    labels = [None, None]
    with tf.variable_scope("Discriminator", reuse=reuse) as scope:
        out_xx = z_generators.z_generator12(in_x, labels, DIM, None, is_training,image_size, reuse)
        # out_x = tf.layers.dense(out_xx, 512)
        # out_x = nonlinearity(out_x)
        # out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        # out_x = tf.layers.dense(out_x, 512)
        # out_x = nonlinearity(out_x)
        # out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        # out_x = tf.layers.dense(out_x, 64)
        # ml1 = tf.reduce_mean((out_x - in_z)**2, 1)

        out_x = tf.layers.dense(out_xx, 512)
        out_x = nonlinearity(out_x)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.dense(out_x, 512)
        out_x = nonlinearity(out_x)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.dense(out_x, 64)


        out_x = tf.layers.dense(out_xx, 512)
        out_x = nonlinearity(out_x)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.dense(out_x, 512)
        out_x = nonlinearity(out_x)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.dense(out_x, 64)
        # ml1 = layers.gaussian_rank1.GaussianRank1(64)((out_x, in_z))
        # ml2 = layers.gaussian_rank1.GaussianRank1(64)((out_x, in_z))
        out_c = tf.layers.dense(out_x, 1)
    return out_c, out_x