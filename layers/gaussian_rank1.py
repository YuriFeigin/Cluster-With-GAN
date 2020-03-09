import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base


class GaussianRank1(base.Layer):

    def __init__(self, z_len):
        super(GaussianRank1, self).__init__()
        self.z_len = z_len
        self.fc_mu = tf.keras.layers.Dense(z_len, kernel_initializer='zeros')
        self.fc_sqrt_diag_precision_mat = tf.keras.layers.Dense(z_len, kernel_initializer='zeros',
                                                                bias_initializer='ones')
        self.fc_rank_precision_mat = tf.keras.layers.Dense(z_len, kernel_initializer='zeros') # initial kernel with zeros cause to local minima (unstale) (no learning in this case)

    def mahalanobis_distance(self, inputs, return_decomposed_precision=False):
        x, z = inputs
        mu = self.fc_mu(x)
        sqrt_diag_precision_mat = self.fc_sqrt_diag_precision_mat(x)
        rank_precision_mat = self.fc_rank_precision_mat(x)
        z = z - mu
        z = tf.reduce_sum((z*sqrt_diag_precision_mat)**2, 1) + tf.reduce_sum(z*rank_precision_mat, 1)**2
        if return_decomposed_precision:
            return z, sqrt_diag_precision_mat, rank_precision_mat
        else:
            return z

    def call(self, inputs, training=None, **kwargs):
        mahalanobis_distance, sqrt_diag_precision_mat, rank_precision_mat = self.mahalanobis_distance(inputs, True)
        log_prob = - 0.5 * self.z_len * tf.math.log(2 * np.pi) + \
            0.5 * tf.math.log(
            (1 + tf.reduce_sum((rank_precision_mat / sqrt_diag_precision_mat) ** 2, 1)) *\
            tf.math.reduce_prod(sqrt_diag_precision_mat ** 2, 1)) + \
            -0.5 * mahalanobis_distance
        return log_prob


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    num_samples_tot = 10000
    mean = [1, 2, 3, 4, 5]
    rank_precision = np.zeros((5, 5))
    rank_precision[0, :] = [1, 0.5, 1.3, 0.1, 1]
    cov = np.linalg.inv(np.matmul(rank_precision.T, rank_precision) + np.diag([4, 5, 3, 2, 3]))
    samples = np.random.multivariate_normal(mean, cov, num_samples_tot).astype(np.float32)

    input_g = tf.keras.Input((3, ), num_samples_tot)
    gaussian_tf = GaussianRank1(5)
    model_tf_cpu = tf.keras.Model(input_g, gaussian_tf((input_g, samples)))

    @tf.function
    def model_tf_gpu(x):
        return model_tf_cpu(x)

    logpdf_tf = gaussian_tf((tf.zeros((num_samples_tot, 3)), samples))

    # gael 1 (high error when gmm_tf initialize with default)
    logpdf = multivariate_normal.logpdf(samples, mean, cov)
    logpdf_tf = model_tf_gpu(tf.zeros((num_samples_tot, 3)))
    results1 = np.mean(abs(logpdf_tf - logpdf))
    print('gael 1 (high error when gmm_tf initialize with default) = {}'.format(results1))

    # gael 2 (GMM optimization)
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    for i in range(2000):
        with tf.GradientTape() as tape:
            logpdf_tf = model_tf_gpu(np.zeros((num_samples_tot, 3), np.float32))
            log_prob = -tf.reduce_mean(logpdf_tf)
            print(log_prob)
        grads = tape.gradient(log_prob, model_tf_cpu.trainable_variables)
        opt.apply_gradients(zip(grads, model_tf_cpu.trainable_variables))
    results2 = np.mean(abs(logpdf_tf - logpdf))
    print('gael 2 (GMM optimization) = {}'.format(results2))

    logpdf_tf = gaussian_tf((tf.zeros((num_samples_tot, 3)), samples))
