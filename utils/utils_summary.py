import tensorflow as tf
import numpy as np


class summary_collection():
    def __init__(self,name):
        self._collection = []

    def add_summary_image(self,in_,name):
        s = tf.summary.image(name,in_)
        self._collection.append(s)
    def add_summary_image1(self, in_, num_imgs, name):
        # side shown is the size of the concatenated image in x and y axes
        side_shown = int(np.sqrt(num_imgs))
        shape = in_.get_shape().as_list()
        image_size = shape[1]
        shown_x = tf.transpose(
            tf.reshape(
                in_[:(side_shown * side_shown), :, :, :],
                [side_shown, image_size * side_shown, image_size, shape[3]]),
            [0, 2, 1, 3])
        shown_x = tf.transpose(
            tf.reshape(
                shown_x,
                [1, image_size * side_shown, image_size * side_shown, shape[3]]),
            [0, 2, 1, 3]) * 128. + 128.
        s = tf.summary.image(
            name,
            tf.cast(shown_x, tf.uint8),
            max_outputs=1)
        self._collection.append(s)

    def add_summary_image2(self,in_1,in_2,num_imgs,name):
        shape = in_1.get_shape().as_list()
        image_size = shape[1]
        side_shown = int(np.sqrt(num_imgs))
        side_shown_h = int(side_shown/2)
        num_samples = int(side_shown**2/2)
        shown_x1 = tf.reshape(in_1[:num_samples, :, :, :],[side_shown_h, image_size * side_shown, image_size, shape[3]])
        shown_x2 = tf.reshape(in_2[:num_samples, :, :, :],[side_shown_h, image_size * side_shown, image_size, shape[3]])
        shown_x = tf.concat([shown_x1,shown_x2],2)
        shown_x = tf.transpose(shown_x,[0, 2, 1, 3])
        shown_x = tf.transpose(
            tf.reshape(
                shown_x,
                [1, image_size * side_shown, image_size * side_shown, shape[3]]),
            [0, 2, 1, 3]) * 128. + 128.
        s = tf.summary.image(
            name,
            tf.cast(shown_x, tf.uint8),
            max_outputs=1)
        self._collection.append(s)

    def add_summary_scalar(self,in_,name):
        s = tf.summary.scalar(name, in_)
        self._collection.append(s)

    def add_summary(self,s):
        self._collection.append(s)

    def get_summary(self):
        return self._collection

    def add_collection(self,summary_collection):
        self._collection.extend(summary_collection.get_summary())

