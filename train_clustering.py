import os
import threading
import collections
import argparse
import logging
import time
from datetime import datetime
import random
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import tensorflow as tf

import load_data
import models.ALI.model1 as model1
import models.ALI.model2 as model2
import models.ALI.ALI_orig_celeba as ALI_orig_celeba
import models.ALI.ALI_orig_cifar10 as ALI_orig_cifar10
import utils.utils_summary as utils_summary
import utils.utils as utils

cluster_sz = [1, 5, 10]

def calc_cluster(sess,all_latent,labels,clustering_algo,global_step,summary_writers,
                                     summary_op,ph_ACC,ph_NMI,ph_ARI,logging):
    ind_labeld_imgs = labels != -1
    labels = labels[ind_labeld_imgs]
    info_str = 'Calculate Clustering'
    for i, hist_len in enumerate(cluster_sz):
        if len(all_latent) >= hist_len:
            y_pred = clustering_algo.fit_predict(np.concatenate(all_latent[-hist_len:], 1))[ind_labeld_imgs]
            acc = utils.ACC(labels, y_pred)[0]
            nmi = metrics.normalized_mutual_info_score(labels, y_pred, 'geometric')
            ari = metrics.adjusted_rand_score(labels, y_pred)
            summary_str = sess.run(summary_op, {ph_ACC: acc, ph_NMI: nmi, ph_ARI: ari})  #
            summary_writers[i].add_summary(summary_str, global_step)
            summary_writers[i].flush()
            info_str += '\nHistory length: {:5d}  ,ACC:{:.3f} , NMI:{:.3f}, ARI{:.3f}'.format(hist_len, acc, nmi, ari)
    logging.info(info_str)


def main(args, logging):
    # get frequently argument
    batch_size = args.batch_size
    z_len = args.z_len
    log_iter = args.log_iter
    max_iter = args.max_iter
    tensorboard_log = args.tensorboard_log
    save_image_iter = args.save_image_iter
    calc_cluster_flag = args.calc_cluster
    cluster_sample = args.cluster_sample

    # prepare for saving latent space
    os.makedirs(os.path.join(args.log_dir, 'latent'), exist_ok=True)
    max_hist = max(cluster_sz)
    latent_samples_iter = np.arange(0, max_iter + 1, args.cluster_sample)[:, np.newaxis] - \
                          np.arange(0, max_hist * args.cluster_gap, args.cluster_gap)[np.newaxis, :]
    latent_samples_iter = np.unique(latent_samples_iter)
    latent_samples_iter = latent_samples_iter[latent_samples_iter >= 0]
    latent_samples_iter = list(latent_samples_iter)
    latent_samples_iter.append(-1) # add dummy iteration
    latent_queue = collections.deque(maxlen=max_hist)

    # choose model
    all_models = {'model1': model1, 'model2': model2, 'ALI_orig_celeba': ALI_orig_celeba,
                  'ALI_orig_cifar10': ALI_orig_cifar10}
    model = all_models.get(args.architecture)

    # prepare data
    dataset_train = load_data.Load(args.dataset, args.train_on, shuffle=True, batch_size=batch_size,
                                   img_size=args.img_size)
    next_element_train = dataset_train.get_imgs_next()
    dataset_eval = load_data.Load(args.dataset, 'all', shuffle=False, batch_size=500, img_size=args.img_size)
    next_element_eval = dataset_eval.get_full_next()
    image_size = [dataset_train.img_size, dataset_train.img_size, next_element_train.shape.as_list()[-1]]

    # define inputs
    input_x = tf.placeholder(tf.float32, [batch_size, ] + image_size)
    input_x_eval = tf.placeholder(tf.float32, [None] + image_size)
    sam_z = tf.placeholder(tf.float32, [args.batch_size, z_len])

    # data augmentation
    imgs_real = input_x + tf.random_uniform(shape=[batch_size] + image_size, minval=0., maxval=1.)  # dequantize
    imgs_real = imgs_real / 128. - 1
    imgs_real = tf.image.random_flip_left_right(imgs_real)

    # network
    x_gen = model.x_generator(sam_z, args.dim_decoder, is_training=True, image_size=image_size, reuse=False)
    z_gen = model.z_generator(imgs_real, z_len, args.dim_encoder, is_training=True, image_size=image_size, reuse=False)

    # lr prior
    imgs_real_lr = tf.image.flip_left_right(imgs_real)
    z_gen_lr = model.z_generator(imgs_real_lr, z_len, args.dim_encoder, is_training=True, image_size=image_size,
                                 reuse=True)

    imgs_concat = tf.concat([imgs_real, x_gen], 0)
    z_concat = tf.concat([z_gen, sam_z], 0)
    t_d = model.discriminator(imgs_concat, z_concat, args.dim_discriminator, is_training=True, image_size=image_size,
                              reuse=False)
    p1, q1 = tf.split(t_d, 2)

    t_d = model.discriminator(imgs_concat, z_concat, args.dim_discriminator, is_training=False, image_size=image_size,
                              reuse=True)
    p2, q2 = tf.split(t_d, 2)

    # cost function
    disc_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p1, labels=tf.ones_like(p1)))
    disc_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q1, labels=tf.zeros_like(q1)))
    disc_loss = (disc_real + disc_fake) / 2
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=q2, labels=tf.ones_like(q2)))
    enc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=p2, labels=tf.zeros_like(p2)))
    enc_gen_loss = (enc_loss + gen_loss) / 2

    z1_loss = -1
    if args.aug > 0:
        z1_loss = tf.reduce_mean(tf.reduce_sum((z_gen - z_gen_lr) ** 2, 1))
        enc_gen_loss += args.aug * z1_loss

    if args.alice:
        lamb = 1 / 1000
        x_rec = model.x_generator(z_gen, args.dim_decoder, is_training=True, image_size=image_size, reuse=True)
        z_rec = model.z_generator(x_gen, z_len, args.dim_encoder, is_training=True, image_size=image_size, reuse=True)
        x_rec_loss = tf.reduce_mean(tf.abs(imgs_real - x_rec))
        z_rec_loss = tf.reduce_mean(tf.abs(sam_z - z_rec))
        enc_gen_loss += lamb * x_rec_loss + lamb * z_rec_loss

    # optimizer
    var = tf.trainable_variables()
    x_gen_var = [v for v in var if 'Decoder' in v.name]
    z_gen_var = [v for v in var if 'Encoder' in v.name]
    disc_var = [v for v in var if 'Discriminator' in v.name]
    gen_save = tf.train.Saver([v for v in tf.global_variables() if 'Decoder' in v.name])
    enc_save = tf.train.Saver([v for v in tf.global_variables() if 'Encoder' in v.name])
    gen_opt = tf.train.AdamOptimizer(learning_rate=args.lr * 5, beta1=0.5, beta2=0.999)
    disc_opt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.5, beta2=0.999)
    gen_gv = gen_opt.compute_gradients(enc_gen_loss, var_list=x_gen_var + z_gen_var)
    disc_gv = disc_opt.compute_gradients(disc_loss, var_list=disc_var)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    # for saving latent space
    t_input_x_eval = input_x_eval / 128. - 1
    z_eval = model.z_generator(t_input_x_eval, z_len, args.dim_encoder, is_training=False, image_size=image_size,
                               reuse=True)

    # save images
    x_rec = model.x_generator(z_gen, args.dim_decoder, is_training=True, image_size=image_size, reuse=True)
    z = np.random.normal(size=(100, z_len)).astype(np.float32)
    z = tf.Variable(z, False)
    x_gen_fix = model.x_generator(z, args.dim_decoder, is_training=True, image_size=image_size, reuse=True)

    if tensorboard_log:
        summary1 = utils_summary.summary_collection('col1')
        summary2 = utils_summary.summary_collection('col2')
        summary_cluster = utils_summary.summary_collection('col3')
        if calc_cluster_flag:
            ph_ACC = tf.placeholder(tf.float32)
            ph_NMI = tf.placeholder(tf.float32)
            ph_ARI = tf.placeholder(tf.float32)
            clustering_algo = KMeans(n_clusters=dataset_eval.num_classes, precompute_distances=True, n_jobs=1)
            with tf.name_scope('cluster'):
                summary_cluster.add_summary_scalar(ph_ACC, 'ACC')
                summary_cluster.add_summary_scalar(ph_NMI, 'NMI')
                summary_cluster.add_summary_scalar(ph_ARI, 'ARI')
        with tf.name_scope('losses'):
            summary1.add_summary_scalar(disc_real, 'disc_real')
            summary1.add_summary_scalar(disc_fake, 'disc_fake')
            summary1.add_summary_scalar(disc_loss, 'disc_loss')
            summary1.add_summary_scalar(enc_gen_loss, 'enc_gen_loss')
            summary1.add_summary_scalar(gen_loss, 'gen_loss')
            summary1.add_summary_scalar(enc_loss, 'enc_loss')
            summary1.add_summary_scalar(z1_loss, 'z1_loss')
            summary1.add_summary_scalar(tf.math.sqrt(tf.reduce_mean(gen_gv[len(x_gen_var)-2][0]**2)), 'gen_grad')
            summary1.add_summary_scalar(tf.math.sqrt(tf.reduce_mean(gen_gv[len(gen_gv)-2][0]**2)), 'enc_grad')
            summary1.add_summary_scalar(tf.math.sqrt(tf.reduce_mean(disc_gv[0][0]**2)), 'disc_grad')
        summary2.add_summary_image2(imgs_real, x_rec, 12 ** 2, 'Input')
        summary2.add_summary_image1(x_gen_fix, args.batch_size, 'Sam')
        summary2.add_collection(summary1)
        summary_op_1 = tf.summary.merge(summary1.get_summary())
        summary_op_2 = tf.summary.merge(summary2.get_summary())
        summary_op_cluster = tf.summary.merge(summary_cluster.get_summary())


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(init)
        # enc_save.restore(sess, tf.train.latest_checkpoint('/root/Documents/Cluster-With-GAN/results/clustering30/cifar10'))
        # gen_save.restore(sess, tf.train.latest_checkpoint('/root/Documents/Cluster-With-GAN/results/clustering30/cifar10'))
        # -- create tensorboard summary writers -- #
        if tensorboard_log:
            summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'tb_summary'), graph=sess.graph)
            summary_writer_cluster = []
            if calc_cluster_flag:
                for Len in cluster_sz:
                    summary_writer_cluster.append(tf.summary.FileWriter(os.path.join(args.log_dir,'tb_Cluster'+str(Len))))
        save_latent_iter = latent_samples_iter[0]
        latent_ind = 0
        it = -1
        ep = -1
        global_step = -1
        cluster_thread = None
        while global_step <= max_iter:  # for epoch
            dataset_train.init_dataset(sess)
            ep += 1
            it_per_epoch = it_in_epoch if it != -1 else -1
            it_in_epoch = 0
            while global_step <= max_iter:  # for iter in epoch
                try:
                    global_step += 1
                    it += 1
                    it_in_epoch += 1

                    # -- train network -- #
                    x = sess.run(next_element_train)
                    z = np.random.normal(size=(batch_size, z_len))
                    start_time = time.time()
                    d_loss, _ = sess.run([disc_loss, disc_train_op], {input_x: x, sam_z: z})
                    for i in range(1):
                        g_loss, _, _ = sess.run([enc_gen_loss, gen_train_op, update_ops], {input_x: x, sam_z: z})
                    duration = time.time() - start_time

                    # -- save log -- #
                    if global_step % log_iter == 0:
                        examples_per_sec = batch_size / float(duration)
                        info_str = '{}: Epoch: {:3d} ({:5d}/{:5d}), global_setp {:6d}, d_loss = {:.2f},g_loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)'.format(
                            datetime.now(), ep, it_in_epoch, it_per_epoch, global_step, d_loss, g_loss,
                            examples_per_sec, duration)
                        logging.info(info_str)
                        print('\r', info_str, end='', flush=True)
                        if tensorboard_log:
                            summary_str = sess.run(summary_op_1, {input_x: x, sam_z: z})
                            summary_writer.add_summary(summary_str, global_step)

                    # -- save latent space to queue-- #
                    if global_step == save_latent_iter:
                        dataset_eval.init_dataset(sess)
                        info_str = 'saving latent iter: ' + str(global_step)
                        logging.info(info_str)
                        print('\r', info_str, end='', flush=True)
                        latent_eval = []
                        label_eval = []
                        while True:
                            try:
                                t_x, t_l = sess.run(next_element_eval)
                                latent_eval.append(z_eval.eval({input_x_eval: t_x}))
                                label_eval.append(t_l)
                            except tf.errors.OutOfRangeError:
                                break
                        latent_eval = np.concatenate(latent_eval, 0)
                        label_eval = np.concatenate(label_eval, 0)
                        latent_queue.append(latent_eval)
                        latent_ind += 1
                        save_latent_iter = latent_samples_iter[latent_ind]

                    # -- calc clustering -- #
                    if global_step % cluster_sample == 0 and calc_cluster_flag:
                        if cluster_thread is not None:
                            cluster_thread.join()
                        latent_list = list(latent_queue)
                        cluster_args = (sess,latent_list,label_eval,clustering_algo,global_step,summary_writer_cluster,
                                     summary_op_cluster,ph_ACC,ph_NMI,ph_ARI,logging)
                        cluster_thread = threading.Thread(target=calc_cluster, args=cluster_args)
                        cluster_thread.start()


                    # -- save images -- #
                    if tensorboard_log and global_step % save_image_iter == 0:
                        summary_str = sess.run(summary_op_2, {input_x: x, sam_z: z})
                        summary_writer.add_summary(summary_str, global_step)
                        summary_writer.flush()

                    if global_step % 100000 == 0 or global_step >= 450000:
                        gen_save.save(sess, os.path.join(args.log_dir, 'gen-model'), global_step=global_step)
                        enc_save.save(sess, os.path.join(args.log_dir, 'enc-model'), global_step=global_step)

                except tf.errors.OutOfRangeError:
                    break
        # save last latent space to disk
        for latents, step in zip(latent_queue, latent_samples_iter[-max_hist-1:-1]):
            np.savez(os.path.join(args.log_dir, 'latent', 'latent' + str(step) + '.npz'),
                     latent=latents)
        cluster_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['celeba', 'cifar10', 'cifar100', 'stl10'], type=str, help='choose dataset')
    parser.add_argument('--train_on', default='all', choices=['train', 'test', 'all'], type=str,
                        help='on which images to train')
    parser.add_argument('--img_size', default=None, type=int, help='the seed of the network initial')
    parser.add_argument('log_dir', type=str, help='where to save all logs')
    parser.add_argument('--architecture', default='model2',
                        choices=['model1', 'model2', 'ALI_orig_celeba', 'ALI_orig_cifar10'],
                        type=str, help='maximum iteration until stop')
    parser.add_argument('--max_iter', default=500000, type=int, help='maximum iteration until stop')
    parser.add_argument('--seed', default=-1, type=int, help='the seed of the network initial')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help='')
    parser.add_argument('--dim_decoder', default=128, type=int, help='decoder dimension')
    parser.add_argument('--dim_encoder', default=128, type=int, help='encoder dimension')
    parser.add_argument('--dim_discriminator', default=128, type=int, help='discriminator dimension')
    parser.add_argument('--z_len', default=64, type=int, help='length of the encoder latent space')
    parser.add_argument('--save_image_iter', default=5000, type=int, help='number of iteration to save images')
    parser.add_argument('--save_images', default=True, type=bool, help='save images')
    parser.add_argument('--log_iter', default=100, type=int, help='number of iteration to save log')
    parser.add_argument('--alice', default=False, type=bool, help='use ALI conditional entropy')
    parser.add_argument('--aug', default=0, type=float,
                        help='weight on the constrain of latent space augmentation')  # 0.001
    parser.add_argument('--tensorboard_log', default=True, type=bool, help='create tensorboard logs')
    parser.add_argument('--gpu', default=0, type=int, help='use gpu number')
    parser.add_argument('--calc_cluster', default=True, action='store_true', help='calculate clustering curves')
    parser.add_argument('--cluster_gap', default=5000, type=int, help='number of iteration to save latent space')
    parser.add_argument('--cluster_sample', default=5000, type=int, help='number of iteration to save latent space')

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
