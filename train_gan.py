"""CT-WGAN ResNet for CIFAR-10"""
"""highly based on the GP-GAN : https://github.com/igul222/improved_wgan_training """

import os
import logging
import random
import argparse
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import models.GAN.model1 as model1
import models.GAN.model2 as model2
import load_data
import utils.utils_summary as utils_summary
import utils.inception_score as inception_score
import utils.ndb as ndb


def main(args, logging):
    # get frequently argument
    batch_size = args.batch_size
    log_iter = args.log_iter
    max_iter = args.max_iter
    tensorboard_log = args.tensorboard_log
    N_CRITIC = args.N_CRITIC
    CALC_INCEPTION = args.calc_is_and_fid
    CALC_NDB = args.calc_ndb
    INCEPTION_FREQUENCY = args.INCEPTION_FREQUENCY
    NDB_FREQUENCY = args.NDB_FREQUENCY

    batch_size_gen = args.GEN_BS_MULTIPLE * batch_size

    # choose model
    all_models = {'model1': model1, 'model2': model2}
    model = all_models.get(args.architecture)
    
    with tf.variable_scope("Train_Data_Input", reuse=False) as scope:
        dataset_train = load_data.Load(args.dataset, args.train_on, shuffle=True, batch_size=batch_size,img_size=args.img_size)
    next_element_train = dataset_train.get_full_next()
    image_size = [dataset_train.img_size,dataset_train.img_size,next_element_train[0].shape.as_list()[-1]]


    if args.label == 'unsup':
        n_labels = None
        fake_labels1 = tf.placeholder(tf.int32,None) # false variable
        fake_labels2 = tf.placeholder(tf.int32,None) # false variable
        fixed_imgs_len = 10 ** 2
        fixed_noise = tf.constant(np.random.normal(size=(fixed_imgs_len, args.z_len)).astype(np.float32))
        fixed_labels = fake_labels2 = tf.placeholder(tf.int32,None) # false variable
        fake_labels_100 = None
    else:
        if args.label == 'clustering':
            n_labels = args.n_clusters
            dataset_train.set_clustring(args.clustering_path, n_labels, args.n_cluster_hist)
        elif args.label == 'sup':
            n_labels = dataset_train.num_classes
        fake_labels1 = tf.cast(tf.random_uniform([batch_size]) * n_labels, tf.int32)
        fake_labels2 = tf.cast(tf.random_uniform([batch_size_gen]) * n_labels, tf.int32)
        fixed_imgs_len = n_labels ** 2
        fixed_noise = tf.constant(np.random.normal(size=(fixed_imgs_len, args.z_len)).astype(np.float32))
        fixed_labels = tf.constant(np.array(list(range(0, n_labels)) * n_labels, dtype=np.int32))

        # fake_labels_100 = tf.cast(tf.random_uniform([100])*n_labels, tf.int32)
        prob = dataset_train.get_label_dist()
        fake_labels_100 = tf.py_func(np.random.choice, [np.arange(n_labels), 100, True, prob], tf.int64)
        fake_labels_100.set_shape(100)

    
    _iteration = tf.placeholder(tf.int32, shape=None)
    input_x = tf.placeholder(tf.float32, shape=[batch_size] + image_size)
    all_real_labels = tf.placeholder(tf.int32, shape=[batch_size])

    # data augmentation
    all_real_data = input_x + tf.random_uniform(shape=[batch_size] + image_size, minval=0., maxval=1.)  # dequantize
    all_real_data = all_real_data / 128. - 1

    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    gen_costs = []
    gen_acgan_costs = []

    all_fake_data = model.Generator(batch_size, args.DIM_G, args.z_len, [all_real_labels, n_labels],
                                    is_training=True, image_size=image_size, reuse=False)

    # Discriminator
    real_and_fake_data = tf.concat([all_real_data, all_fake_data], axis=0)
    real_and_fake_labels = tf.concat([all_real_labels, fake_labels1], axis=0)
    disc_all, disc_all_2, disc_all_acgan = model.Discriminator(real_and_fake_data, args.DIM_D, [real_and_fake_labels, n_labels],
                                                               is_training=True, image_size=image_size, reuse=False)

    disc_real, disc_fake = tf.split(disc_all, 2)
    disc_real_2, disc_fake_2 = tf.split(disc_all_2, 2)

    # Discriminator for Consistency Term (CT)
    disc_all_, disc_all_2_, disc_all_acgan_ = model.Discriminator(real_and_fake_data, args.DIM_D, [real_and_fake_labels, n_labels],
                                                                  is_training=True, image_size=image_size, reuse=True)
    disc_real_, disc_fake_ = tf.split(disc_all_, 2)
    disc_real_2_, disc_fake_2_ = tf.split(disc_all_2_, 2)

    # wasserstein distance
    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
    # gradient penalty
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    differences = all_fake_data - all_real_data
    interpolates = all_real_data + (alpha * differences)
    gp_disc = model.Discriminator(interpolates, args.DIM_D, [all_real_labels, n_labels],
                                  is_training=True, image_size=image_size, reuse=True)[0]
    gradients = tf.gradients(gp_disc, [interpolates])[0]  # same dropout rate
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = 10.0 * tf.reduce_mean((slopes - 1.) ** 2)
    disc_costs.append(gradient_penalty)
    # consistency term
    CT = args.LAMBDA_2 * tf.square(disc_real - disc_real_)
    CT += args.LAMBDA_2 * 0.1 * tf.reduce_mean(tf.square(disc_real_2 - disc_real_2_), axis=[1])
    CT_ = tf.maximum(CT - args.Factor_M, 0.0 * (CT - args.Factor_M))
    CT_ = tf.reduce_mean(CT_)
    disc_costs.append(CT_)

    # train the generator
    x = model.Generator(batch_size_gen, args.DIM_G, args.z_len, [fake_labels2,n_labels], is_training=True, image_size=image_size, reuse=True)
    disc_fake, disc_fake_2, disc_fake_acgan = model.Discriminator(x, args.DIM_D, [fake_labels2,n_labels],
                                                                  is_training=True, image_size=image_size, reuse=True)
    gen_costs.append(-tf.reduce_mean(disc_fake))

    # build the loss function
    disc_wgan = tf.add_n(disc_costs)
    if n_labels is not None:
        disc_all_acgan_real, disc_all_acgan_fake = tf.split(disc_all_acgan, 2)
        disc_acgan_costs.append(tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan_real, labels=all_real_labels)))

        disc_acgan_accs.append(tf.reduce_mean(
            tf.cast(tf.equal(
                tf.to_int32(tf.argmax(disc_all_acgan_real, axis=1)), all_real_labels), tf.float32)
        ))

        disc_acgan_fake_accs.append(tf.reduce_mean(
            tf.cast(tf.equal(
                tf.to_int32(tf.argmax(disc_all_acgan_fake, axis=1)), all_real_labels), tf.float32)
        ))
        gen_acgan_costs.append(tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels2)
        ))
        disc_acgan = tf.add_n(disc_acgan_costs)
        disc_acgan_acc = tf.add_n(disc_acgan_accs)
        disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs)
        disc_cost = disc_wgan + (args.ACGAN_SCALE * disc_acgan)
        gen_cost = tf.add_n(gen_costs) + (args.ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs)))
    else:
        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan
        gen_cost = tf.add_n(gen_costs)

    if args.DECAY:
        decay = tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / max_iter))
    else:
        decay = 1.

    var = tf.trainable_variables()
    gen_var = [v for v in var if 'Generator' in v.name]
    disc_var = [v for v in var if 'Discriminator' in v.name]

    gen_opt = tf.train.AdamOptimizer(learning_rate=args.lr * decay, beta1=0.0, beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=args.lr * decay, beta1=0.0, beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=gen_var)
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_var)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    if tensorboard_log:
        tf_inception_m1 = tf.placeholder(tf.float32, shape=None)
        tf_inception_std1 = tf.placeholder(tf.float32, shape=None)
        tf_inception_m2 = tf.placeholder(tf.float32, shape=None)
        tf_inception_std2 = tf.placeholder(tf.float32, shape=None)
        tf_fid = tf.placeholder(tf.float32, shape=None)
        tf_ndb = tf.placeholder(tf.float32, shape=None)
        tf_ndb_js = tf.placeholder(tf.float32, shape=None)
        summary1 = utils_summary.summary_collection('col1')
        summary2 = utils_summary.summary_collection('col2')
        summary3 = utils_summary.summary_collection('col3')
        summary4 = utils_summary.summary_collection('col4')
        with tf.name_scope('disc'):
            summary1.add_summary_scalar(disc_cost, 'disc_cost')
            summary1.add_summary_scalar(disc_wgan, 'disc_wgan')
            summary1.add_summary_scalar(disc_acgan, 'disc_acgan')
        with tf.name_scope('ACGAN'):
            summary1.add_summary_scalar(disc_acgan_acc, 'acc_real')
            summary1.add_summary_scalar(disc_acgan_fake_acc, 'acc_fake')
        with tf.name_scope('gen'):
            summary1.add_summary_scalar(gen_cost, 'gen_cost')
        with tf.name_scope('inception'):
            summary3.add_summary_scalar(tf_inception_m1, 'incep_mean')
            summary3.add_summary_scalar(tf_inception_std1, 'incep_std')
            summary3.add_summary_scalar(tf_inception_m2, 'incep_mean')
            summary3.add_summary_scalar(tf_inception_std2, 'incep_std')
            summary3.add_summary_scalar(tf_fid, 'fid')
            summary4.add_summary_scalar(tf_ndb, 'ndb')
            summary4.add_summary_scalar(tf_ndb_js, 'ndb_js')

        # Function for generating samples
        fixed_noise_samples = model.Generator(fixed_imgs_len, args.DIM_G, args.z_len, [fixed_labels, n_labels], is_training=True,
                                              image_size=image_size, reuse=True, noise=fixed_noise)
        summary2.add_summary_image1(fixed_noise_samples, fixed_imgs_len, 'Sam')


        summary_op_1 = tf.summary.merge(summary1.get_summary())
        summary_op_2 = tf.summary.merge(summary2.get_summary())
        summary_op_3 = tf.summary.merge(summary3.get_summary())
        summary_op_4 = tf.summary.merge(summary4.get_summary())

    # Function for calculating inception score
    samples_100 = model.Generator(100, args.DIM_G, args.z_len, [fake_labels_100, n_labels], is_training=True, image_size=image_size,
                                  reuse=True)

    with tf.Session() as sess:
        def get_samples(n):
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(sess.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
            return all_samples

        dataset_train._init_dataset()
        TrainData = dataset_train.load_sub_imgs(80000)
        if CALC_INCEPTION:
            inception_score.update_fid_mean(TrainData)
        if CALC_NDB:
            ndb_model = ndb.NDB(TrainData, max_dims=2000, semi_whitening=True)

        sess.run(tf.global_variables_initializer())

        # saver = tf.train.Saver(tf.global_variables())

        # ckpt_state = tf.train.get_checkpoint_state(os.path.join(log_dir,'ckpt'))
        # if ckpt_state and ckpt_state.model_checkpoint_path:
        #     print("Loading file %s" % ckpt_state.model_checkpoint_path)
        #     saver.restore(sess, ckpt_state.model_checkpoint_path)

        summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'summary'), graph=sess.graph)
        os.makedirs(os.path.join(args.log_dir, 'data'), exist_ok=True)
        _disc_cost = 0
        _disc_wgan = 0
        it = -1
        ep = -1
        global_step = -1
        best_IS1 = -1
        while global_step<=max_iter:  # for epoch
            dataset_train.init_dataset(sess)
            ep += 1
            it_per_epoch = it_in_epoch if it != -1 else -1
            it_in_epoch = 0
            while global_step<=max_iter:  # for iter in epoch
                try:
                    start_time = time.time()
                    _ = sess.run([gen_train_op], feed_dict={_iteration: global_step})
                    for i in range(N_CRITIC):
                        x, y = sess.run(next_element_train)
                        y = np.squeeze(y)
                        if n_labels is not None:
                            _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = sess.run(
                                [disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                                feed_dict={input_x: x, all_real_labels: y, _iteration: global_step})
                        else:
                            _disc_cost, _ = sess.run([disc_cost, disc_train_op],
                                                     feed_dict={input_x: x, all_real_labels: y,
                                                                _iteration: global_step})
                    duration = time.time() - start_time

                    if global_step % log_iter == 0:
                        examples_per_sec = batch_size / float(duration)
                        info_str = ('{}: Epoch: {:3d} ({:5d}/{:5d}), global_setp {:5d}, disc_cost = {:.2f},disc_wgan = {:.2f} '
                                      '({:.1f} examples/sec; {:.3f} '
                                      'sec/batch)').format(datetime.now(), ep, it_in_epoch, it_per_epoch, global_step, _disc_cost, _disc_wgan,
                            examples_per_sec, duration)
                        logging.info(info_str)
                        print('\r', info_str, end='', flush=True)

                        if global_step % 1000 == 0:
                            summary_str = sess.run(summary_op_2, {input_x: x, all_real_labels: y})
                        else:
                            summary_str = sess.run(summary_op_1, {input_x: x, all_real_labels: y})
                        summary_writer.add_summary(summary_str, global_step)
                        summary_writer.flush()

                    if tensorboard_log and CALC_INCEPTION and global_step % INCEPTION_FREQUENCY == 0:
                        print('\r', 'calculate inception scrore and FID', end='', flush=True)
                        samples1 = get_samples(50000)
                        inception_score1_m, inception_score1_s, fid1 = inception_score.calc_scores(samples1)
                        info_str = 'IS_mean: {:6.3f} , IS_std: {:6.3f} , fid: {:6.3f}'.format(inception_score1_m,
                                                                                              inception_score1_s, fid1)
                        logging.info(info_str)
                        if inception_score1_m>best_IS1:
                            best_IS1 = inception_score1_m
                            samples1 = get_samples(50000)
                            inception_score2_m, inception_score2_s, fid2 = inception_score.calc_scores(samples1)
                            info_str = 'IS_mean2: {:6.3f} , IS_std2: {:6.3f} , fid2: {:6.3f}'.format(inception_score1_m,
                                                                                    inception_score1_s,fid1)
                            logging.info(info_str)
                        else:
                            inception_score2_m, inception_score2_s = 0,0
                        summary_str = sess.run(summary_op_3, {tf_inception_m1: inception_score1_m,
                                                              tf_inception_std1: inception_score1_s,
                                                              tf_inception_m2: inception_score2_m,
                                                              tf_inception_std2: inception_score2_s,
                                                              tf_fid: fid1})
                        summary_writer.add_summary(summary_str, global_step)
                        summary_writer.flush()

                    if tensorboard_log and CALC_NDB and global_step % NDB_FREQUENCY == 0:
                        print('\r', 'calculate NDB', end='', flush=True)
                        samples = get_samples(20000)
                        results = ndb_model.evaluate(samples)
                        info_str = 'ndb: {:6.3f} , ndb_js: {:6.3f}'.format(results['NDB'],results['JS'])
                        logging.info(info_str)
                        summary_str = sess.run(summary_op_4, {tf_ndb: results['NDB'],tf_ndb_js: results['JS']})
                        summary_writer.add_summary(summary_str, global_step)
                        summary_writer.flush()

                    # # Save the model checkpoint periodically.
                    # if global_step % checkpoint_iter == 0 and checkpoint_save:
                    #     checkpoint_path = os.path.join(log_dir,'ckpt', 'model')
                    #     saver.save(
                    #         sess,
                    #         checkpoint_path,
                    #         global_step=global_step)
                    global_step += 1
                    it += 1
                    it_in_epoch += 1
                except tf.errors.OutOfRangeError:
                    break


if __name__ == "__main__":

    # checkpoint_iter = 20000
    # checkpoint_save = True

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['celeba', 'cifar10', 'cifar100', 'stl10'], type=str, help='choose dataset')
    parser.add_argument('log_dir', type=str, help='where to save all logs')
    parser.add_argument('label', choices=['unsup', 'sup', 'clustering'], type=str, help='choose dataset')
    parser.add_argument('--clustering_path', type=str, help='clustering path')
    parser.add_argument('--n_clusters',default=40, type=int, help='number of clusters')
    parser.add_argument('--n_cluster_hist',default=10, type=int, help='number of latent space to be used from history')
    parser.add_argument('--train_on', default='train', choices=['train', 'test', 'all'], type=str,
                        help='on which images to train')
    parser.add_argument('--img_size', default=None, type=int, help='the seed of the network initial')
    parser.add_argument('--architecture', default='model2', choices=['model1', 'model2', 'model2'],
                        type=str, help='maximum iteration until stop')
    parser.add_argument('--max_iter', default=200000, type=int, help='maximum iteration until stop')
    parser.add_argument('--seed', default=-1, type=int, help='the seed of the network initial')
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--DECAY', default=True, type=bool, help='Whether to decay LR over learning')
    parser.add_argument('--batch_size', default=64, type=int, help='')
    parser.add_argument('--DIM_G', default=128, type=int, help='Generator dimensionality')
    parser.add_argument('--DIM_D', default=128, type=int, help='Critic dimensionality')
    parser.add_argument('--z_len', default=128, type=int, help='length of the encoder latent space')
    parser.add_argument('--ACGAN_SCALE', default=0.2, type=float,
                        help='How to scale the critic ACGAN loss relative to WGAN loss')
    parser.add_argument('--ACGAN_SCALE_G', default=0.02, type=float,
                        help='How to scale generator ACGAN loss relative to WGAN loss')
    parser.add_argument('--N_CRITIC', default=5, type=int, help='Critic steps per generator steps')
    parser.add_argument('--GEN_BS_MULTIPLE', default=2, type=int,
                        help='Generator batch size, as a multiple of batch_size')
    parser.add_argument('--Factor_M', default=0, type=float, help='factor M')
    parser.add_argument('--LAMBDA_2', default=2, type=float, help='parameter LAMBDA2')
    parser.add_argument('--INCEPTION_FREQUENCY', default=5000, type=int, help=' How frequently to calculate NDB')
    parser.add_argument('--NDB_FREQUENCY', default=5000, type=int, help=' How frequently to calculate Inception score')
    parser.add_argument('--calc_is_and_fid', default=True, type=bool, help=' whether to calculate ndb ')
    parser.add_argument('--calc_ndb', default=False, type=bool, help=' whether to calculate inseption score and fid ')
    parser.add_argument('--save_images', default=True, type=bool, help='save images')
    parser.add_argument('--log_iter', default=20, type=int, help='number of iteration to save log')
    parser.add_argument('--tensorboard_log', default=True, type=bool, help='create tensorboard logs')
    parser.add_argument('--gpu', default=0, type=int, help='use gpu number')

    args = parser.parse_args()
    os.makedirs(args.log_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed != -1:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)

    logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'), filemode='a', format='%(message)s',
                        level=logging.DEBUG)

    # save args to log file
    for arg, value in sorted(vars(args).items()):
        log_str = "Argument {}: {}".format(arg, value)
        logging.info(log_str)
        print(log_str)
    main(args, logging)
