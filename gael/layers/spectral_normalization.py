import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations


def max_singular_value(w_reshaped, u, ip):
    u_hat = u
    v_hat = None

    for _ in range(ip):
        v_ = tf.matmul(w_reshaped, u_hat)
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(tf.transpose(w_reshaped), v_hat)
        u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.transpose(u_hat), tf.matmul(tf.transpose(w_reshaped), v_hat))
    return sigma, u_hat


class SNConv2D(layers.Layer):

    def __init__(self, filters, kernel_size, padding='VALID', activation=None, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', ip=1):
        super(SNConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.ip = ip

    def build(self, input_shape):
        self.w = self.add_weight('kernel', shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.b = self.add_weight('bias', shape=(self.filters,),
                                 initializer=self.bias_initializer,
                                 trainable=True)
        self.u = self.add_weight('u', shape=(self.filters, 1),
                                 initializer='random_normal',
                                 trainable=False)

    def w_bar(self, training):
        w_reshaped = tf.reshape(self.w, [-1, self.filters])
        sigma, u_hat = max_singular_value(w_reshaped, self.u, self.ip)
        if training:
            with tf.control_dependencies([sigma]):
                self.u.assign(u_hat)
        w_norm = self.w / sigma
        return w_norm

    def call(self, inputs, training=None, **kwargs):
        w_norm = self.w_bar(training)
        o = tf.nn.conv2d(inputs, w_norm, 1, self.padding)
        o = tf.nn.bias_add(o, self.b)
        o = self.activation(o)
        return o


class SNLinear(layers.Layer):

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', ip=1):
        super(SNLinear, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.ip = ip
        self.Flatten = layers.Flatten()

    def build(self, input_shape):
        self.w = self.add_weight('weight', shape=(np.prod(input_shape[1:]), self.units),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        if self.use_bias:
            self.b = self.add_weight('bias', shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     trainable=True)
        self.u = self.add_weight('u', shape=(self.units, 1),
                                 initializer='random_normal',
                                 trainable=False)

    def w_bar(self, training):
        w_reshaped = tf.reshape(self.w, [-1, self.units])
        sigma, u_hat = max_singular_value(w_reshaped, self.u, self.ip)

        if training:
            with tf.control_dependencies([sigma]):
                self.u.assign(u_hat)
        w_norm = self.w / sigma
        return w_norm

    def call(self, inputs, training=None, **kwargs):
        w_norm = self.w_bar(training)
        inputs = self.Flatten(inputs)
        o = tf.matmul(inputs, w_norm)
        if self.use_bias:
            o += self.b
        o = self.activation(o)
        return o


class SNEmbedID(layers.Layer):

    def __init__(self, n_classes, weight_initializer='glorot_uniform', ip=1):
        super(SNEmbedID, self).__init__()
        self.n_classes = n_classes
        self.weight_initializer = weight_initializer
        self.ip = ip

    def build(self, input_shape):
        self.w = self.add_weight('weight', shape=(self.n_classes, input_shape[-1]),
                                 initializer=self.weight_initializer,
                                 trainable=True)

        self.u = self.add_weight('u', shape=(self.n_classes, 1),
                                 initializer='random_normal',
                                 trainable=False)

    def w_bar(self, training):
        w_reshaped = tf.transpose(self.w)
        sigma, u_hat = max_singular_value(w_reshaped, self.u, self.ip)
        if training:
            with tf.control_dependencies([sigma]):
                self.u.assign(u_hat)
        w_norm = self.w / sigma
        return w_norm

    def call(self, inputs, training=None, **kwargs):
        labels = inputs
        w_norm = self.w_bar(training)
        x = tf.nn.embedding_lookup(w_norm, labels)
        return x
