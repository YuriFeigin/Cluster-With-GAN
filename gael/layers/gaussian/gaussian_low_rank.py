import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base


class GaussianLowRank(base.Layer):

    def __init__(self, z_len, rank):
        super(GaussianLowRank, self).__init__()
        self.z_len = z_len
        self.rank = rank
        self.fc_mu = tf.keras.layers.Dense(z_len, kernel_initializer='zeros')
        self.fc_diag_precision_mat = tf.keras.layers.Dense(z_len, kernel_initializer='zeros',
                                                           bias_initializer='ones')
        self.fc_rank_precision_mat = tf.keras.layers.Dense(self.rank * z_len, kernel_initializer='zeros')

    def build(self, input_shape):
        self.const = tf.zeros((input_shape[0][0], self.z_len-self.rank, self.z_len))

    def get_decomposed_precision_mat(self, x):
        diag_precision_mat = self.fc_diag_precision_mat(x)
        rank_precision_mat = self.fc_rank_precision_mat(x)
        rank_precision_mat = tf.reshape(rank_precision_mat, (-1, self.rank, self.z_len))
        decomposed_percision_mat = tf.linalg.diag(diag_precision_mat) + \
            tf.concat([rank_precision_mat, self.const], 1)
        return decomposed_percision_mat

    def mahalanobis_distance(self, inputs, return_decomposed_precision=False):
        x, z = inputs
        mu = self.fc_mu(x)
        decomposed_precision_mat = self.get_decomposed_precision_mat(x)
        z = z - mu
        z = tf.reduce_sum(tf.reduce_sum(tf.expand_dims(z, -1) * decomposed_precision_mat, -2) ** 2, -1)
        if return_decomposed_precision:
            return z, decomposed_precision_mat
        else:
            return z

    def call(self, inputs, training=None, **kwargs):
        mahalanobis_distance, decomposed_precision_mat = self.mahalanobis_distance(inputs, True)
        log_prob = - 0.5 * self.z_len * tf.math.log(2 * np.pi) + \
            0.5 * tf.linalg.logdet(
            tf.matmul(decomposed_precision_mat, decomposed_precision_mat, transpose_a=True)) + \
            -0.5 * mahalanobis_distance
        return log_prob


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    num_samples_tot = 10000
    mean = [1, 2, 3, 4, 5]
    rank_precision = np.zeros((5, 5))
    rank_precision[0, :] = [1, 0.5, 1.3, 0.1, 1]
    precision_matrix = rank_precision + np.diag([4, 5, 3, 2, 3])
    cov = np.linalg.inv(np.matmul(precision_matrix.T, precision_matrix))
    samples = np.random.multivariate_normal(mean, cov, num_samples_tot).astype(np.float32)

    input_g = tf.keras.Input((3, ), num_samples_tot)
    gaussian_tf = GaussianLowRank(5, 1)
    model_tf_cpu = tf.keras.Model(input_g, gaussian_tf((input_g, samples)))

    @tf.function
    def model_tf_gpu(x):
        return model_tf_cpu(x)

    logpdf_tf = gaussian_tf((tf.zeros((num_samples_tot, 3)), samples))

    # test 1 (high error when gmm_tf initialize with default)
    logpdf = multivariate_normal.logpdf(samples, mean, cov)
    logpdf_tf = model_tf_gpu(tf.zeros((num_samples_tot, 3)))
    results1 = np.mean(abs(logpdf_tf - logpdf))
    print('test 1 (high error when gmm_tf initialize with default) = {}'.format(results1))

    # test 2 (GMM optimization)
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    for i in range(2000):
        with tf.GradientTape() as tape:
            logpdf_tf = model_tf_gpu(np.zeros((num_samples_tot, 3), np.float32))
            log_prob = -tf.reduce_mean(logpdf_tf)
            print(log_prob)
        grads = tape.gradient(log_prob, model_tf_cpu.trainable_variables)
        opt.apply_gradients(zip(grads, model_tf_cpu.trainable_variables))
    results2 = np.mean(abs(logpdf_tf - logpdf))
    print('test 2 (GMM optimization) = {}'.format(results2))

    logpdf_tf = gaussian_tf((tf.zeros((num_samples_tot, 3)), samples))
