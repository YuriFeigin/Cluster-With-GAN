import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import gael.layers.conditional_layers as conditional_layers
from gael.layers.common import *


class _ResBlock(layers.Layer, Common):

    def __init__(self, filters, activation, kernel_size=3, n_classes=None):
        super(_ResBlock, self).__init__()
        self.filters = filters
        self.activation = activations.get(activation)
        self.n_classes = n_classes
        self.hidden_channels = filters  # can be different
        self.ksize = kernel_size

        # initializer1 = tf.keras.initializers.VarianceScaling(mode='fan_avg', distribution="uniform") # todo add make peoblem why?
        # initializer2 = tf.keras.initializers.VarianceScaling(2., mode='fan_avg', distribution="uniform")

        self.c1 = layers.Conv2D(filters, kernel_size, padding='same')
        self.c2 = layers.Conv2D(filters, kernel_size, padding='same')
        if n_classes is None:
            self.b1 = layers.BatchNormalization()
            self.b2 = layers.BatchNormalization()
        else:
            self.b1 = conditional_layers.CondBatchNormalization(n_classes)
            self.b2 = conditional_layers.CondBatchNormalization(n_classes)
        self.c_sc = layers.Conv2D(filters, 1, padding='same')  # learnable skip connection


class OptimizedResBlock(_ResBlock):

    def __init__(self, filters, activation, kernel_size=3, n_classes=None):
        super(OptimizedResBlock, self).__init__(filters, activation, kernel_size, n_classes)
        self.down_sample = layers.AveragePooling2D(2)

    def call(self, inputs, **kwargs):
        x_in, labels = inputs
        x = self.c1(x_in)
        x = self.activation(x)
        x = self.c2(x)
        x = self.down_sample(x)

        sc = self.down_sample(x_in)
        sc = self.c_sc(sc)
        return x + sc


class ResBlockDown(_ResBlock):

    def __init__(self, filters, activation, kernel_size=3, n_classes=None):
        super(ResBlockDown, self).__init__(filters, activation, kernel_size, n_classes)
        self.down_sample = layers.AveragePooling2D(2)

    def call(self, inputs, **kwargs):
        x_in, labels = inputs
        x = self.b1(x_in, **kwargs) if labels is None else self.b1((x_in, labels), **kwargs)
        x = self.activation(x)
        x = self.c1(x)
        x = self.b2(x, **kwargs) if labels is None else self.b2((x, labels), **kwargs)
        x = self.activation(x)
        x = self.c2(x)
        x = self.down_sample(x)

        sc = self.c_sc(x_in)
        sc = self.down_sample(sc)

        return x + sc


class ResBlock(_ResBlock):

    def __init__(self, filters, activation, kernel_size=3):
        super(ResBlock, self).__init__(filters, activation, kernel_size)
        self.down_sample = layers.AveragePooling2D(2)

    def call(self, inputs, y=None, **kwargs):
        x_in, labels = inputs
        x = self.activation(x_in)
        x = self.c1(x)
        x = self.c2(x)

        return x + x_in


class ResBlockUp(_ResBlock):

    def __init__(self, filters, activation, kernel_size=3, n_classes=None):
        super(ResBlockUp, self).__init__(filters, activation, kernel_size, n_classes)
        self.upsampling = layers.UpSampling2D()

    def call(self, inputs, **kwargs):
        x_in, labels = inputs
        x = self.b1(x_in, **kwargs) if labels is None else self.b1((x_in, labels), **kwargs)
        x = self.activation(x)
        x = self.upsampling(x)
        x = self.c1(x)
        x = self.b2(x, **kwargs) if labels is None else self.b2((x, labels), **kwargs)
        x = self.activation(x)
        x = self.c2(x)

        sc = self.upsampling(x_in)
        sc = self.c_sc(sc)

        return x + sc