import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations


class CondBatchNormalization(layers.Layer):

    def __init__(self, n_classes):
        super(CondBatchNormalization, self).__init__()
        self.n_classes = n_classes
        self.b1 = layers.BatchNormalization(center=False, scale=False, trainable=True)

    def build(self, input_shape):
        self.offset_m = self.add_weight('offset', shape=(self.n_classes, input_shape[0][-1]),
                                 initializer=tf.initializers.zeros,
                                 trainable=True)
        self.scale_m = self.add_weight('scale', shape=(self.n_classes, input_shape[0][-1]),
                                 initializer=tf.initializers.ones,
                                 trainable=True)

    def call(self, inputs, training=None, **kwargs):
        x, labels = inputs
        offset = tf.nn.embedding_lookup(self.offset_m, labels)
        scale = tf.nn.embedding_lookup(self.scale_m, labels)
        x = self.b1(x, training=training)
        x = x * scale[:, None, None, :] + offset[:, None, None, :]
        return x