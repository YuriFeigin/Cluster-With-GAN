import tensorflow as tf

LEAK = 0.1


def maxout(in_, n_piece):
    shape = in_.get_shape().as_list()
    out = tf.reshape(in_, shape[:-1] + [int(shape[-1] / n_piece), 2])
    out = tf.reduce_max(out, -1)
    return out


def x_generator(in_,DIM, is_training,image_size, reuse):
    shape = in_.get_shape().as_list()
    with tf.variable_scope("Decoder", reuse=reuse) as scope:
        out = tf.reshape(in_,[shape[0],1,1,shape[1]])
        out = tf.layers.conv2d_transpose(out, filters=512, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d_transpose(out, filters=256, kernel_size=7, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d_transpose(out, filters=256, kernel_size=5, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d_transpose(out, filters=128, kernel_size=7, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d_transpose(out, filters=64, kernel_size=2, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d(out, filters=3, kernel_size=1, strides=1)
        out = tf.nn.tanh(out)
        return out


def z_generator(in_,z_len,DIM_En, is_training,image_size, reuse):
    shape = in_.get_shape().as_list()
    with tf.variable_scope("Encoder", reuse=reuse) as scope:
        out = tf.layers.conv2d(in_, filters=64, kernel_size=2, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d(out, filters=128, kernel_size=7, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d(out, filters=256, kernel_size=5, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d(out, filters=256, kernel_size=7, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.conv2d(out, filters=512, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.02)
        out = tf.layers.flatten(out)
        out1 = tf.layers.dense(out,z_len)
        sigma = tf.layers.dense(out, z_len)
        if is_training:
            noise = tf.random_normal([shape[0], z_len])
            out1 = out1 + sigma * noise
        return out1


def discriminator(in_x, in_z,DIM, is_training,image_size, reuse):
    with tf.variable_scope("Discriminator", reuse=reuse) as scope:
        out_x = tf.layers.dropout(in_x, 0.2, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=64, kernel_size=2, strides=1)
        out_x = tf.nn.leaky_relu(out_x, 0.02)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=128, kernel_size=7, strides=2)
        out_x = tf.nn.leaky_relu(out_x, 0.02)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=256, kernel_size=5, strides=2)
        out_x = tf.nn.leaky_relu(out_x, 0.02)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=256, kernel_size=7, strides=2)
        out_x = tf.nn.leaky_relu(out_x, 0.02)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=512, kernel_size=4, strides=1)
        out_x = tf.nn.leaky_relu(out_x, 0.02)
        out_x = tf.layers.flatten(out_x)

        out_z = tf.layers.dropout(in_z, 0.2, training=is_training)
        out_z = tf.layers.dense(out_z, 1024)
        out_z = tf.nn.leaky_relu(out_z, 0.02)
        out_z = tf.layers.dropout(out_z, 0.2, training=is_training)
        out_z = tf.layers.dense(out_z, 1024)
        out_z = tf.nn.leaky_relu(out_z, 0.02)

        concatenated = tf.concat([out_x, out_z], axis=-1)
        out_c = tf.layers.dropout(concatenated, 0.2, training=is_training)
        out_c = tf.layers.dense(out_c, 2048)
        out_c = tf.nn.leaky_relu(out_c, 0.02)
        out_c = tf.layers.dropout(out_c, 0.2, training=is_training)
        out_c = tf.layers.dense(out_c, 2048)
        out_c = tf.nn.leaky_relu(out_c, 0.02)
        out_c = tf.layers.dropout(out_c, 0.2, training=is_training)
        out_c = tf.layers.dense(out_c, 1)
        return out_c