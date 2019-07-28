import tensorflow as tf
from models.Normalizing import *

def NormalizeG(name, inputs,is_training,labels):
    with tf.variable_scope(name) as scope:
        if labels[1] is None:
            return batch_norm(inputs,name,is_training=is_training)
        else:
            return cond_batchnorm(inputs, name,is_training=is_training,labels=labels[0],n_labels=labels[1])

# def NormalizeD(name, inputs,labels=None):
#     with tf.variable_scope(name) as scope:
#         if NORMALIZATION_D:
#             if not CONDITIONAL:
#                 labels = None
#             output = tf.transpose(inputs, [0, 3, 1, 2])
#             output = lib.ops.layernorm.Layernorm(name, [1, 2, 3], output, labels=labels, n_labels=n_labels)
#             return tf.transpose(output, [0, 2, 3, 1])
#         return inputs

def nonlinearity(x):
    return tf.nn.relu(x)


def UpsampleConv(inputs,output_dim, filter_size,name):
    with tf.variable_scope(name) as scope:
        shape = inputs.get_shape().as_list()
        output = tf.image.resize_nearest_neighbor(images=inputs,size=[shape[1]*2,shape[2]*2])
        output = tf.layers.conv2d(output,output_dim,filter_size,padding='same',name='conv')
    return output

def ConvMeanPool(inputs,output_dim, filter_size,name):
    with tf.variable_scope(name) as scope:
        output = tf.layers.conv2d(inputs, output_dim, filter_size, padding='same', name='Conv')
        output = tf.layers.average_pooling2d(output,2,2)
    return output

def MeanPoolConv(inputs,output_dim, filter_size,name):
    with tf.variable_scope(name) as scope:
        output = tf.layers.average_pooling2d(inputs,2,2)
        output = tf.layers.conv2d(output, output_dim, filter_size, padding='same', name='Conv')
    return output

def OptimizedResBlockDisc1(inputs,DIM_D):
    with tf.variable_scope("OptimizedResBlock1") as scope:
        shortcut = MeanPoolConv(inputs, output_dim=DIM_D, filter_size=1,name='MeanPoolConv')

        output = tf.layers.conv2d(inputs,DIM_D,3, padding='same', name='Conv1')
        output = nonlinearity(output)
        output = ConvMeanPool(output, output_dim=DIM_D, filter_size=3,name='ConvMeanPool')
    return shortcut + output

def ResidualBlock_Up(name, output_dim, filter_size,is_training, inputs, labels=[None,None]):
    with tf.variable_scope(name) as scope:
        shortcut = UpsampleConv(inputs, output_dim=output_dim, filter_size=1, name='Shortcut')

        output = NormalizeG('N1', inputs,is_training, labels=labels)
        output = nonlinearity(output)
        output = UpsampleConv(output,output_dim, filter_size, name='UpConv1')
        output = NormalizeG('N2', output,is_training, labels=labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output, output_dim, filter_size,padding='same', name='ConvOut')
    return shortcut + output

def ResidualBlock_Down(name, output_dim, filter_size, is_training, Normalization,inputs, labels=[None,None]):
    with tf.variable_scope(name) as scope:
        shortcut = ConvMeanPool(inputs, output_dim=output_dim, filter_size=1,name='shortcut')
        output = inputs
        
        if Normalization:
            output = NormalizeG('N1', inputs, is_training, labels=labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output, output_dim, filter_size, padding='same', name='Conv1')
        
        if Normalization:
            output = NormalizeG('N2', output,is_training, labels=labels)
        output = nonlinearity(output)
        output = ConvMeanPool(output, output_dim, filter_size=filter_size,name='ConvMeanPool')

    return shortcut + output

# def ResidualBlock_Down_Enc(name, output_dim, filter_size,is_training, inputs):
#     with tf.variable_scope(name) as scope:
#         shortcut = ConvMeanPool(inputs, output_dim=output_dim, filter_size=1,name='shortcut')
# 
#         output = NormalizeG('N1', inputs,is_training, labels=[None,None])
#         output = nonlinearity(output)
#         output = tf.layers.conv2d(output, output_dim, filter_size, padding='same', name='Conv1')
#         output = NormalizeG('N2', output,is_training, labels=[None,None])
#         output = nonlinearity(output)
#         output = ConvMeanPool(output, output_dim, filter_size=filter_size,name='ConvMeanPool')
# 
#     return shortcut + output

def ResidualBlock(name, output_dim, filter_size, inputs, labels=None):
    with tf.variable_scope(name) as scope:
        if output_dim == inputs.get_shape()[3]:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = tf.layers.conv2d(inputs, output_dim, 1, padding='same', name='Shortcut')
        
        output = inputs
        # output = NormalizeD('N1', inputs, labels=labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output, output_dim, filter_size, padding='same', name='Conv1')
        # output = NormalizeD('N2', output, labels=labels)
        output = nonlinearity(output)
        output = tf.layers.conv2d(output, output_dim, filter_size, padding='same', name='Conv2')

    return shortcut + output




