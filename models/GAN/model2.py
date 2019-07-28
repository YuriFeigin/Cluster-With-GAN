import numpy as np
import tensorflow as tf
from models.ResidualBlocks import *
import models.z_generators as z_generators
import models.x_generators as x_generators


def Generator(n_samples,DIM_G,z_len, labels,is_training,image_size, reuse, noise=None):
    with tf.variable_scope("Generator", reuse=reuse) as scope:
        if noise is None:
            noise = tf.random_normal([n_samples, z_len])
        output = x_generators.x_generator1(noise,labels,DIM_G,is_training,image_size, reuse)
        return output

def Discriminator(inputs,DIM_D, labels,is_training,image_size, reuse):
        with tf.variable_scope("Discriminator", reuse=reuse) as scope:
            output2 = z_generators.z_generator13(inputs, labels,None, DIM_D, False,is_training, image_size, reuse)
            output_wgan = tf.layers.dense(output2, 1, name='Output')
            output_wgan = tf.reshape(output_wgan, [-1])  # conrresponding to D
            if labels[1] is None:
                return output_wgan, output2, None  # two layers' of output
            else:
                output_acgan = tf.layers.dense(output2, labels[1], name='ACGANOutput')
                return output_wgan, output2, output_acgan
