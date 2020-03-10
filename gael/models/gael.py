import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
import gael.layers.residual_blocks.wgan_gp as blocks
from gael.layers.common import Common as LayerCommon
from gael.models.common import Common as ModelCommon
from gael.utils import clustering


class Model(ModelCommon, LayerCommon):
    def __init__(self, tensorboard_path, z_dim, ch=128, bottom_width=4, activation='relu', n_classes=None):
        self.z_dim = z_dim
        self.ch = ch
        self.bottom_width = bottom_width
        self.activation = activations.get(activation)
        self.n_classes = n_classes
        self.num_images_tensorboard = 100

        self.const_z = np.random.normal(size=(self.num_images_tensorboard , z_dim)).astype(np.float32)
        self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_path)

        self.generator = Generator(ch, bottom_width, activation, n_classes)
        self.encoder = Encoder(z_dim, ch, activation, n_classes)
        self.discriminator = Discriminator(z_dim, ch, activation, n_classes)

        self.opt_gen = tf.keras.optimizers.Adam(1e-4 * 5, 0.5, 0.999)
        self.opt_dis = tf.keras.optimizers.Adam(1e-4, 0.5, 0.999)

    @staticmethod
    def input_augmentation(images):
        images = images / 128. - 1
        images = tf.image.random_flip_left_right(images)
        return images

    def discriminator_loss(self, inputs, step):
        imgs_real, z, labels = inputs
        imgs_real = self.input_augmentation(imgs_real)

        x_gen = self.generator((z, labels), training=True)
        z_gen = self.encoder((imgs_real, labels), training=True)

        real_fake_cost, z_reconstruct_cost = self.discriminator((tf.concat([imgs_real, x_gen], 0),
                                                                tf.concat([z_gen, z], 0), None), training=True)
        p1, q1 = tf.split(real_fake_cost, 2)
        ml2_1, ml2_2 = tf.split(z_reconstruct_cost, 2)

        disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p1, labels=tf.ones_like(p1)))
        disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q1, labels=tf.zeros_like(q1)))
        disc_ml_z = tf.reduce_mean((ml2_2-z)**2)

        disc_loss = (disc_real + disc_fake) / 2 + disc_ml_z / 2

        with self.tensorboard_writer.as_default():
            with tf.name_scope('losses_disc'):
                tf.summary.scalar('disc_real', disc_real, step)
                tf.summary.scalar('disc_fake', disc_fake, step)
                tf.summary.scalar('disc_ml_z', disc_ml_z, step)
                tf.summary.scalar('disc_loss', disc_loss, step)
        return disc_loss

    @tf.function
    def train_discriminator(self, inputs, step):
        with tf.GradientTape() as tape:
            disc_loss = self.discriminator_loss(inputs, step)
        gradients_of_discriminator = tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.opt_dis.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return disc_loss

    def generator_encoder_loss(self, inputs, step):
        imgs_real, z, labels = inputs
        imgs_real = self.input_augmentation(imgs_real)

        x_gen = self.generator((z, labels), training=True)
        z_gen = self.encoder((imgs_real, labels), training=True)

        real_fake_cost, z_reconstruct_cost = self.discriminator((tf.concat([imgs_real, x_gen], 0),
                                                                tf.concat([z_gen, z], 0), labels), training=False)
        p1, q1 = tf.split(real_fake_cost, 2)
        ml2_1, ml2_2 = tf.split(z_reconstruct_cost, 2)

        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q1, labels=tf.ones_like(q1)))
        enc_ml_z = tf.reduce_mean((ml2_1-z_gen)**2)

        gen_enc_loss = gen_loss + enc_ml_z / 2

        with self.tensorboard_writer.as_default():
            with tf.name_scope('losses_gen_enc'):
                tf.summary.scalar('gen_loss', gen_loss, step) # todo get summary to trains function gen + enc
                tf.summary.scalar('enc_ml_z', enc_ml_z, step)
                tf.summary.scalar('gen_enc_loss', gen_enc_loss, step)
        return gen_enc_loss

    @tf.function
    def train_generator_encoder(self, inputs, step):
        with tf.GradientTape() as tape:
            gen_enc_loss = self.generator_encoder_loss(inputs, step)
        train_var = self.generator.trainable_variables + self.encoder.trainable_variables
        gradients_of_discriminator = tape.gradient(gen_enc_loss, train_var)
        self.opt_dis.apply_gradients(zip(gradients_of_discriminator, train_var))
        return gen_enc_loss

    # @tf.function
    def images_summary(self, step):
        with self.tensorboard_writer.as_default():
            x_gen = self.generator((self.const_z, None), training=False)
            with self.tensorboard_writer.as_default():
                tf.summary.image('Sampling', self.convert_batch_images_to_one_image(x_gen), step)

    # @tf.function
    def reconstruction_images_summary(self, images, step):
        with self.tensorboard_writer.as_default():
            images = images / 128. - 1
            z_gen = self.encoder((images, None), training=False)
            x_gen = self.generator((z_gen, None), training=False)
            with self.tensorboard_writer.as_default():
                tf.summary.image('Reconstruct',
                                 self.convert_batch_reconstructed_images_to_one_image(images, x_gen), step)

    def eval_clustering(self, dataset, step):
        @tf.function
        def eval_batch(batch, labels=None):
            z_gen = self.encoder((batch, labels), training=False)
            return z_gen

        latent = []
        labels = []
        for t_x, t_labels in dataset:
            latent.append(eval_batch(t_x))
            labels.append(t_labels)
        latent = np.concatenate(latent)
        labels = np.squeeze(np.concatenate(labels))
        results = clustering.calc_cluster(latent, labels, 10) #todo change number of labels
        with self.tensorboard_writer.as_default():
            with tf.name_scope('cluster'):
                tf.summary.scalar('ACC', results['ACC'], step)
                tf.summary.scalar('NMI', results['NMI'], step)
                tf.summary.scalar('ARI', results['ARI'], step)


class Generator(tf.keras.Model, LayerCommon):
    def __init__(self, ch, bottom_width, activation, n_classes):
        super(Generator, self).__init__()
        self.ch = ch
        self.bottom_width = bottom_width
        self.activation = activations.get(activation)
        self.n_classes = n_classes
        self.dense1 = layers.Dense(bottom_width ** 2 * ch * 4)
        self.reshape2 = layers.Reshape([bottom_width, bottom_width, ch * 4])
        self.block_up3 = blocks.ResBlockUp(ch * 4, activation, n_classes=n_classes)
        self.block_up4 = blocks.ResBlockUp(ch * 2, activation, n_classes=n_classes)
        self.block_up5 = blocks.ResBlockUp(ch * 1, activation, n_classes=n_classes)
        self.batch_norm6 = layers.BatchNormalization()  # todo add conditional batch norm
        self.conv7 = layers.Conv2D(3, 3, padding='SAME', activation='tanh')

    def call(self, inputs, training=None, mask=None):
        x, labels = inputs
        x = self.dense1(x)
        x = self.reshape2(x)
        x = self.block_up3((x, labels))
        x = self.block_up4((x, labels))
        x = self.block_up5((x, labels))
        x = self.batch_norm6(x)
        x = self.activation(x)
        x = self.conv7(x)
        return x


class Encoder(tf.keras.Model, LayerCommon):
    def __init__(self, z_dim, ch, activation, n_classes):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.activation = activations.get(activation)
        self.n_classes = n_classes

        self.conv1 = layers.Conv2D(ch, 3, padding='SAME')
        self.block_down2 = blocks.ResBlockDown(ch * 1, activation, n_classes=n_classes)
        self.block_down3 = blocks.ResBlockDown(ch * 2, activation, n_classes=n_classes)
        self.block_down4 = blocks.ResBlockDown(ch * 4, activation, n_classes=n_classes)
        self.bn5 = layers.BatchNormalization()
        self.flatten6 = layers.Flatten()
        self.dense7 = layers.Dense(z_dim)

    def call(self, inputs, training=None, mask=None):
        x, labels = inputs
        x = self.conv1(x)
        x = self.block_down2((x, labels))
        x = self.block_down3((x, labels))
        x = self.block_down4((x, labels))
        x = self.bn5(x)
        x = self.flatten6(x)
        x = self.dense7(x)
        return x


class Discriminator(tf.keras.Model, LayerCommon):
    def __init__(self, z_dim, ch, activation, n_classes):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.ch = ch
        self.activation = activations.get(activation)
        self.n_classes = n_classes

        self.opt_block1 = blocks.OptimizedResBlock(ch, activation, use_bn=False)
        self.block_down2 = blocks.ResBlockDown(ch, activation, use_bn=False)
        self.dropout1 = layers.Dropout(0.2)
        self.block3 = blocks.ResBlock(ch, activation, use_bn=False)
        self.dropout2 = layers.Dropout(0.5)
        self.block4 = blocks.ResBlock(ch, activation, use_bn=False)
        self.dropout3 = layers.Dropout(0.5)
        # self.global_average_pool5 = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()

        self.dense1_path1 = layers.Dense(512, activation)
        self.dropout1_path1 = layers.Dropout(0.5)
        self.dense2_path1 = layers.Dense(512, activation)
        self.dropout2_path1 = layers.Dropout(0.5)
        self.classifier = layers.Dense(1)

        self.dense1_path2 = layers.Dense(512, activation)
        self.dropout1_path2 = layers.Dropout(0.5)
        self.dense2_path2 = layers.Dense(512, activation)
        self.dropout2_path2 = layers.Dropout(0.5)
        self.out1 = layers.Dense(z_dim)

    def call(self, inputs, training=None, mask=None):
        x, z, labels = inputs
        x = self.opt_block1((x, labels))
        x = self.block_down2((x, labels))
        x = self.dropout1(x)
        x = self.block3((x, labels))
        x = self.dropout2(x)
        x = self.block4((x, labels))
        x = self.dropout3(x)
        x = self.activation(x)
        # x = self.global_average_pool5(x)
        # x = tf.reduce_sum(x, (1, 2))
        x = self.flatten(x)

        x1 = self.dense1_path1(x)
        x1 = self.dropout1_path1(x1)
        x1 = self.dense2_path1(x1)
        x1 = self.dropout2_path1(x1)
        x1 = self.classifier(x1)

        x2 = self.dense1_path2(x)
        x2 = self.dropout1_path2(x2)
        x2 = self.dense2_path2(x2)
        x2 = self.dropout2_path2(x2)
        x2 = self.out1(x2)

        return x1, x2
