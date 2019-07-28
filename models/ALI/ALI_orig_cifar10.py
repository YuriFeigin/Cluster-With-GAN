import tensorflow as tf

LEAK = 0.1


def maxout(in_, n_piece):
    shape = in_.get_shape().as_list()
    out = tf.reshape(in_, shape[:-1] + [int(shape[-1] / n_piece), 2])
    out = tf.reduce_max(out, -1)
    return out


def x_generator(in_,is_training, reuse):
    shape = in_.get_shape().as_list()
    with tf.variable_scope("x_generator", reuse=reuse) as scope:
        out = tf.reshape(in_,[shape[0],1,1,shape[1]])
        out = tf.layers.conv2d_transpose(out, filters=256, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d_transpose(out, filters=128, kernel_size=4, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d_transpose(out, filters=64, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d_transpose(out, filters=32, kernel_size=4, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d_transpose(out, filters=32, kernel_size=5, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=32, kernel_size=1, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=3, kernel_size=1, strides=1)
        out = tf.nn.sigmoid(out)
        return out


def z_generator(in_,is_training, reuse):
    shape = in_.get_shape().as_list()
    with tf.variable_scope("z_generator", reuse=reuse) as scope:
        out = tf.layers.conv2d(in_, filters=32, kernel_size=5, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=64, kernel_size=4, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=128, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=256, kernel_size=4, strides=2)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=512, kernel_size=4, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.conv2d(out, filters=512, kernel_size=1, strides=1)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.leaky_relu(out, 0.1)
        out = tf.layers.flatten(out)
        mu = tf.layers.dense(out,64)
        sigma = tf.layers.dense(out, 64)
        noise = tf.random_normal([shape[0], 64])
        out1 = mu + sigma * noise
        return out1,mu,sigma


def discriminator(in_x, in_z, is_training, reuse):
    with tf.variable_scope("Discriminator", reuse=reuse) as scope:
        out_x = tf.layers.dropout(in_x, 0.2, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=32, kernel_size=5, strides=1)
        out_x = maxout(out_x, 2)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=64, kernel_size=4, strides=2)
        out_x = maxout(out_x, 2)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=128, kernel_size=4, strides=1)
        out_x = maxout(out_x, 2)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=256, kernel_size=4, strides=2)
        out_x = maxout(out_x, 2)
        out_x = tf.layers.dropout(out_x, 0.5, training=is_training)
        out_x = tf.layers.conv2d(out_x, filters=512, kernel_size=4, strides=1)
        out_x = maxout(out_x, 2)
        out_x = tf.layers.flatten(out_x)

        out_z = tf.layers.dropout(in_z, 0.2, training=is_training)
        out_z = tf.layers.dense(out_z, 512)
        out_z = maxout(out_z, 2)
        out_z = tf.layers.dropout(out_z, 0.5, training=is_training)
        out_z = tf.layers.dense(out_z, 512)
        out_z = maxout(out_z, 2)

        concatenated = tf.concat([out_x, out_z], axis=-1)
        out_c = tf.layers.dropout(concatenated, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1024)
        out_c = maxout(out_c, 2)
        out_c = tf.layers.dropout(out_c, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1024)
        out_c = maxout(out_c, 2)
        out_c = tf.layers.dropout(out_c, 0.5, training=is_training)
        out_c = tf.layers.dense(out_c, 1)
        return out_c