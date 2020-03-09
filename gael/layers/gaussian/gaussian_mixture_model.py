import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base


class GMM(base.Layer):

    def __init__(self, n_classes, weight_initializer='glorot_uniform'):
        super(GMM, self).__init__()
        self.n_classes = n_classes
        self.weight_initializer = weight_initializer

    def build(self, input_shape):
        np_shape = input_shape.as_list()
        np_init = np.broadcast_to(np.eye(np_shape[-1]), (self.n_classes, np_shape[-1], np_shape[-1]))
        self.w = self.add_weight('w', shape=(self.n_classes, ),
                                 initializer=tf.ones_initializer(),
                                 trainable=True)
        self.mu = self.add_weight('mu', shape=(self.n_classes, input_shape[-1]),
                                  initializer=tf.random_normal_initializer(0, 0.05),
                                  trainable=True)
        self.decomposed_precision_mat = \
            self.add_weight('decomposed_precision', shape=(self.n_classes, input_shape[-1], input_shape[-1]),
                            initializer=tf.constant_initializer(np_init),
                            trainable=True)

    def mahalanobis_distance(self, inputs):
        z = inputs
        z = tf.expand_dims(z, -2) - self.mu
        z = tf.reduce_sum(tf.reduce_sum(tf.expand_dims(z, -1) * self.decomposed_precision_mat, -2) ** 2, -1)
        return z

    def log_prob_for_each_gmm(self, inputs):
        z = inputs
        z = self.mahalanobis_distance(z)
        w = tf.abs(self.w)
        z = tf.math.log(w/tf.reduce_sum(w)) + \
            - 0.5 * tf.cast(tf.shape(self.mu)[1], tf.float32) * tf.math.log(2 * np.pi) + \
            0.5 * tf.linalg.logdet(tf.matmul(self.decomposed_precision_mat, self.decomposed_precision_mat, transpose_a=True)) + \
            - 0.5 * z
        return z

    def call(self, inputs, training=None, **kwargs):
        z = inputs
        z = self.log_prob_for_each_gmm(z)
        log_prob = tf.math.reduce_logsumexp(z, -1)
        return log_prob

    def update_params(self, w, mu, cov_mat):
        precision_mat = np.linalg.inv(cov_mat)
        decomposed_precision_mat = np.linalg.cholesky(precision_mat)
        self.w.assign(w)
        self.mu.assign(mu)
        self.decomposed_precision_mat.assign(decomposed_precision_mat)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    num_samples_tot = 100000
    weight = [0.1, 0.4, 0.25, 0.25]
    mean = [[4, 1],
            [0, 4],
            [-4, 2],
            [-1, -5]]
    cov = [[[1, 0], [0, 1]],
           [[1, 0.5], [0.5, 1]],
           [[3, 0], [0, 1]],
           [[1, 1], [1, 2]]]
    samples = []
    for i in range(len(weight)):
        num_samples = round(weight[i]*num_samples_tot)
        samples.append(np.random.multivariate_normal(mean[i], cov[i], num_samples))
    samples = np.concatenate(samples, 0).astype(np.float32)

    gmm_tf = GMM(4)
    gmm = GaussianMixture(n_components=4).fit(samples)

    # test 1 (high error when gmm_tf initialize with default)
    results1 = np.mean(abs(gmm_tf(samples) - gmm.score_samples(samples)))
    print('test 1 (high error when gmm_tf initialize with default) = {}'.format(results1))

    # test 2 (GMM optimization)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(2000):
        with tf.GradientTape() as tape:
            log_prob = -tf.reduce_mean(gmm_tf(samples))
        grads = tape.gradient(log_prob, gmm_tf.trainable_variables)
        opt.apply_gradients(zip(grads, gmm_tf.trainable_variables))
    results2 = np.mean(abs(gmm_tf(samples) - gmm.score_samples(samples)))
    print('test 2 (GMM optimization) = {}'.format(results2))

    # test 3 (low error when gmm_tf updates to correct parameters)
    gmm_tf.update_params(weight, mean, cov)
    results2 = np.mean(abs(gmm_tf(samples) - gmm.score_samples(samples)))
    print('test 3 (low error when gmm_tf updates to correct parameters) = {}'.format(results2))