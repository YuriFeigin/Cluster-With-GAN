import logging
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from models import gael
import load_data

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

dataset = 'cifar10'
train_on = 'all'
batch_size = 100
pad_size = 0
img_size = 32
z_dim = 64
log_iter = 100
max_iter = 500000
save_image_iter = 5000
tensorboard_path = 'results'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
imgs = np.concatenate([x_train, x_test]).astype(np.float32)
labels = np.concatenate([y_train, y_test])

dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
dataset = dataset.shuffle(4096)
dataset = dataset.repeat(1000).batch(batch_size, drop_remainder=True)

dataset_eval = tf.data.Dataset.from_tensor_slices((imgs, labels)).batch(1000)

# # prepare data
# dataset_train = load_data.Load(dataset, train_on, shuffle=True, batch_size=batch_size, pad_size=pad_size,
#                                img_size=img_size)
# next_element_train = dataset_train.get_imgs_next()
# dataset_eval = load_data.Load(dataset, 'all', shuffle=False, batch_size=500, pad_size=0, img_size=img_size)
# next_element_eval = dataset_eval.get_full_next()

model = gael.Model(tensorboard_path, z_dim)

it = -1
ep = -1
global_step = -1
while global_step <= max_iter:  # for epoch
    ep += 1
    it_per_epoch = it_in_epoch if it != -1 else -1
    it_in_epoch = 0

    for x, labels in dataset:
        global_step += 1
        it += 1
        it_in_epoch += 1
        tf.summary.experimental.set_step(global_step)

        # -- train network -- #
        z = np.random.normal(size=(batch_size, z_dim)).astype(np.float32)
        start_time = time.time()
        for i in range(1):
            g_loss = model.train_generator_encoder((x, z, None), tf.cast(global_step, tf.int64))
        d_loss = model.train_discriminator((x, z, None), tf.cast(global_step, tf.int64))
        duration = time.time() - start_time

        # -- save log -- #
        if global_step % log_iter == 0:
            examples_per_sec = batch_size / float(duration)
            info_str = '{}: Epoch: {:3d} ({:5d}/{:5d}), global_setp {:6d}, d_loss = {:.2f},g_loss = {:.2f}, ({:.1f} examples/sec; {:.3f} sec/batch)'.format(
                datetime.now(), ep, it_in_epoch, it_per_epoch, global_step, d_loss, g_loss,
                examples_per_sec, duration)
            logging.info(info_str)
            print('\r', info_str, end='', flush=True)
            model.tensorboard_writer.flush()

        # # -- calc clustering -- #
        # if global_step % cluster_sample == 0 and calc_cluster_flag:
        #     if cluster_thread is not None:
        #         cluster_thread.join()
        #     latent_list = list(latent_queue)
        #     cluster_args = (sess, latent_list, label_eval, clustering_algo, global_step, summary_writer_cluster,
        #                     summary_op_cluster, ph_ACC, ph_NMI, ph_ARI, logging)
        #     cluster_thread = threading.Thread(target=calc_cluster, args=cluster_args)
        #     cluster_thread.start()
        #     if global_step > 290000:
        #         gmm.fit(latent_list[-1])

        # -- save images -- #
        if global_step % save_image_iter == 0:
            model.eval_clustering(dataset_eval, tf.cast(global_step, tf.int64))
            model.images_summary(global_step)
            model.reconstruction_images_summary(x, tf.cast(global_step, tf.int64))

        # if global_step % 100000 == 0 or (global_step >= 450000 and global_step % 5000 == 0):
        #     gen_save.save(sess, os.path.join(args.log_dir, 'gen-model'), global_step=global_step)
        #     enc_save.save(sess, os.path.join(args.log_dir, 'enc-model'), global_step=global_step)
