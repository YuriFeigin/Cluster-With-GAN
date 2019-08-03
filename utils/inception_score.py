# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import sys
import warnings
from scipy import linalg

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None
pool3 = None
pool3_mean_real = None
pool3_std_real = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_features(images):
    assert ((images.shape[3]) == 3)
    assert (np.max(images) > 10)
    assert (np.min(images) >= 0.0)
    images = images.astype(np.float32)
    bs = 100
    sess = tf.get_default_session()
    preds = []
    feats = []
    for inp in np.array_split(images, round(images.shape[0] / bs)):
        # sys.stdout.write(".")
        # sys.stdout.flush()
        [feat, pred] = sess.run([pool3, softmax], {'InputTensor:0': inp})
        feats.append(feat.reshape(-1, 2048))
        preds.append(pred)
    feats = np.concatenate(feats, 0)
    preds = np.concatenate(preds, 0)
    return preds, feats


def update_fid_mean(images):
    global pool3_mean_real
    global pool3_std_real
    preds, feats = get_features(images)
    pool3_mean_real = np.mean(feats, axis=0)
    pool3_std_real = np.cov(feats, rowvar=False)


def calc_scores(images, splits=10):
    preds, feats = get_features(images)

    # calc inception
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    inception_m = np.mean(scores)
    inception_s = np.std(scores)

    # fid
    mu2 = np.mean(feats, axis=0)
    sigma2 = np.cov(feats, rowvar=False)
    fid = calculate_frechet_distance(pool3_mean_real, pool3_std_real, mu2, sigma2)

    return inception_m, inception_s, fid


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


# This function is called automatically.
def _init_inception():
    global softmax
    global pool3
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.GFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Import model with a modification in the input tensor to accept arbitrary
        # batch size.
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                      name='InputTensor')
        _ = tf.import_graph_def(graph_def, name='inception_v3',
                                input_map={'ExpandDims:0': input_tensor})
    # Works with an arbitrary minibatch size.
    pool3 = tf.get_default_graph().get_tensor_by_name('inception_v3/pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        if 'inception_v3' in op.name:
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
    w = tf.get_default_graph().get_operation_by_name("inception_v3/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
    softmax = tf.nn.softmax(logits)

_init_inception()
