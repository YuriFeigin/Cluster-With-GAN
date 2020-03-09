import tensorflow as tf


class Common(object):
    @staticmethod
    def convert_batch_images_to_one_image(batch_images):
        shape = tf.shape(batch_images)
        image_size = shape[1]
        side_shown = tf.cast(tf.floor(tf.sqrt(tf.cast(shape[0], tf.float32))), tf.int32)
        shown_x = tf.transpose(
            tf.reshape(
                batch_images[:(side_shown * side_shown), :, :, :],
                [side_shown, image_size * side_shown, image_size, shape[3]]),
            [0, 2, 1, 3])
        shown_x = tf.transpose(
            tf.reshape(
                shown_x,
                [1, image_size * side_shown, image_size * side_shown, shape[3]]),
            [0, 2, 1, 3]) * 128. + 128.
        shown_x = tf.cast(shown_x, tf.uint8)
        return shown_x

    @staticmethod
    def convert_batch_reconstructed_images_to_one_image(batch_real_images, batch_rec_images):
        shape = tf.shape(batch_real_images)
        image_size = shape[1]
        side_shown = tf.cast(tf.floor(tf.sqrt(tf.cast(shape[0], tf.float32)* 2)), tf.int32)
        side_shown_h = tf.cast(side_shown / 2, tf.int32)
        num_samples = tf.cast(side_shown ** 2 / 2, tf.int32)
        shown_x1 = tf.reshape(batch_real_images[:num_samples, :, :, :], [side_shown_h, image_size * side_shown, image_size, shape[3]])
        shown_x2 = tf.reshape(batch_rec_images[:num_samples, :, :, :], [side_shown_h, image_size * side_shown, image_size, shape[3]])
        shown_x = tf.concat([shown_x1, shown_x2], 2)
        shown_x = tf.transpose(shown_x, [0, 2, 1, 3])
        shown_x = tf.transpose(
            tf.reshape(
                shown_x,
                [1, image_size * side_shown, image_size * side_shown, shape[3]]),
            [0, 2, 1, 3]) * 128. + 128.
        shown_x = tf.cast(shown_x, tf.uint8)
        return shown_x
