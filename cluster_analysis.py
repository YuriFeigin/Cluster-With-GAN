import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import logging
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import load_data
import utils.utils as utils


def full(args,x_test,y_test):
    logging.basicConfig(filename=os.path.join(args.data_path, 'full_clustering.log'), filemode='a', format='%(message)s',
                        level=logging.DEBUG)
    try:
        # save args to log file
        for arg, value in sorted(vars(args).items()):
            log_str = "Argument {}: {}".format(arg, value)
            logging.info(log_str)
        time.sleep(60)
        LengthSample = [1, 5, 10]
        MaxLengthSample = np.array(LengthSample).max()
        ph_ACC = tf.placeholder(tf.float32)
        ph_NMI = tf.placeholder(tf.float32)
        ph_ARI = tf.placeholder(tf.float32)

        kmeans = KMeans(n_clusters=dataset_eval.num_classes, precompute_distances=True, n_jobs=32)
        with tf.name_scope('cluster'):
            summary_ACC = tf.summary.scalar('ACC', ph_ACC)
            summary_NMI = tf.summary.scalar('NMI', ph_NMI)
            summary_ARI = tf.summary.scalar('ARI', ph_ARI)
        summary_op = tf.summary.merge_all()
        summary_writers = []
        data_all = []
        CurIter = -1
        y_test = y_test
        IndLabeldImgs = y_test != -1
        y_test = y_test[IndLabeldImgs]
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        last_iter = int(np.floor(args.max_iter / args.latent_iter) * args.latent_iter)
        with tf.Session(config=config) as sess:
            for Len in LengthSample:
                summary_writers.append(tf.summary.FileWriter(os.path.join(args.data_path,'tb_Cluster'+str(Len))))
            while CurIter <= last_iter:  # for epoch
                files = np.array(os.listdir(os.path.join(args.data_path,'latent')))
                IterNumExist = np.array([int(f[6:-4]) for f in files])
                IndAboveCur = IterNumExist > CurIter
                IterNumExist = IterNumExist[IndAboveCur]
                files = files[IndAboveCur]
                if np.any(IndAboveCur):
                    info_str = 'process Iteration : ' + str(CurIter)
                    logging.info(info_str)
                    ind = np.argmin(IterNumExist - CurIter)
                    CurIter = IterNumExist[ind]
                    data = np.load(os.path.join(args.data_path,'latent',files[ind]))
                    data_all.append(data['latent'])
                    if data_all.__len__() > MaxLengthSample:
                        del data_all[0]
                    for i,Len in enumerate(LengthSample):
                        if data_all.__len__() < Len:
                            continue
                        else:
                            kmeans.fit(np.concatenate(data_all[-Len:], 1))
                            y_pred = kmeans.labels_[IndLabeldImgs]
                            ACC = utils.ACC(y_test, y_pred)[0]
                            NMI = metrics.normalized_mutual_info_score(y_test, y_pred,'geometric')
                            ARI = metrics.adjusted_rand_score(y_test, y_pred)
                            summary_str = sess.run(summary_op, {ph_ACC: ACC,ph_NMI:NMI,ph_ARI:ARI})#
                            summary_writers[i].add_summary(summary_str, CurIter)
                            summary_writers[i].flush()
                            info_str = 'History length: {:5d}  ,ACC:{:.3f} , NMI:{:.3f}, ARI{:.3f}'.format(Len,ACC,NMI,ARI)
                            logging.info(info_str)
                else:
                    time.sleep(30)
        logging.info('finish')
    except Exception:
        logging.exception('this is an exception')

def get_latent(args,hist_len):
    files = np.array(os.listdir(os.path.join(args.data_path, 'latent')))
    IterNumExist = np.array([int(f[6:-4]) for f in files])
    argsort = np.argsort(IterNumExist)
    if hist_len > len(IterNumExist):
        return None
    cur_time_steps = IterNumExist[argsort[-hist_len:]]
    print('### running for history length of:', str(hist_len))
    print('using follow time steps:', cur_time_steps)
    data = []
    for i in cur_time_steps:
        data.append(np.load(os.path.join(args.data_path, 'latent', 'latent' + str(i) + '.npz'))['latent'])
    return np.concatenate(data, 1)

def get_clustering(args,hist_len,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=True, n_jobs=32)
    data = get_latent(args, hist_len)
    kmeans.fit(data)
    y_pred = kmeans.labels_
    return y_pred

def final(args,x_test,y_test):
    LengthSample = [1, 5, 10]
    for l in LengthSample:
        y_pred = get_clustering(args,l,dataset_eval.num_classes)
        if y_pred is not None:
            ACC = utils.ACC(y_test, y_pred)[0]
            print('ACC = ',str(ACC))
            NMI = metrics.normalized_mutual_info_score(y_test, y_pred)
            print('NMI = ',str(NMI))
            ARI = metrics.adjusted_rand_score(y_test, y_pred)
            print('ARI = ',str(ARI))

def draw(args,x_test,y_test):
    n_clusters = args.num_clusters
    shape = x_test.shape
    y_pred = get_clustering(args, 2, n_clusters)
    img = np.zeros([n_clusters * shape[1], 10 * shape[2], shape[3]])
    for i in range(n_clusters):
        ind = np.random.permutation(np.nonzero(y_pred == i)[0])[:10]
        img[shape[1] * i:shape[1] * (i + 1), :] = np.reshape(np.transpose(x_test[ind], [1, 0, 2, 3]),
                                                             [shape[1], shape[2] * 10, shape[3]]) / 255
    plt.imshow(img)
    plt.show()

def tsne(args,x_test,y_test):
    ind = np.random.permutation(y_test.shape[0])[:10000]
    x_test = x_test[ind]
    y_test = y_test[ind]
    LOG_DIR = './cache/tsne'
    os.makedirs(LOG_DIR,exist_ok=True)
    spirits_file = 'spirit.png'
    metadata_file = 'metadata.tsv'
    path_for_sprites = os.path.join(LOG_DIR,spirits_file)
    path_for_metadata = os.path.join(LOG_DIR,metadata_file)
    # create sprite image
    img_h = x_test.shape[1]
    img_w = x_test.shape[2]
    n_plots = int(np.ceil(np.sqrt(x_test.shape[0])))

    sprite_image = np.ones((img_h * n_plots, img_w * n_plots,3),np.uint8)

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < x_test.shape[0]:
                this_img = x_test[this_filter]
                sprite_image[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img
    plt.imsave(path_for_sprites, sprite_image)

    with open(path_for_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(y_test):
            f.write("%d\t%d\n" % (index, label))
    latent = get_latent(args, 10)[ind]
    embedding_var = tf.Variable(latent.reshape(latent.shape[0],-1), name='data')
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = metadata_file #'metadata.tsv'
    embedding.sprite.image_path = spirits_file #'mnistdigits.png'
    embedding.sprite.single_image_dim.extend([img_h,img_w])
    projector.visualize_embeddings(summary_writer, config)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='choose data')
    parser.add_argument('mode',choices=['full', 'final', 'draw','tsne'], type=str, help='choose dataset')
    parser.add_argument('--seed', default=-1, type=int, help='the seed of the network initial')
    parser.add_argument('--num_clusters', default=20, type=int, help='number of samples in history')


    args = parser.parse_args()
    # todo check if folder exist

    if args.seed != -1:
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        random.seed(args.seed)
    
    # get dataset params
    with open(os.path.join(args.data_path,'training.log'), 'r') as f:
        i=0
        for line in f:
            if 'Argument dataset' in line:
                args.dataset = line.strip().split()[-1]
                i+=1
            elif 'Argument max_iter' in line:
                args.max_iter = float(line.strip().split()[-1])
                i+=1
            elif 'Argument save_latent_iter' in line:
                args.latent_iter = float(line.strip().split()[-1])
                i+=1
            if i==3:
                break

    # get all images and labels
    dataset_eval = load_data.Load(args.dataset, 'all', shuffle=False,  batch_size=5000, img_size=None)
    next_element_eval = dataset_eval.get_full_next()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        x_test = []
        y_test = []
        dataset_eval.init_dataset(sess)
        while True:
            try:
                t_x, t_y = sess.run(next_element_eval)
                x_test.append(t_x)
                y_test.append(t_y)
            except tf.errors.OutOfRangeError:
                break
        x_test = np.squeeze(np.concatenate(x_test, 0))
        y_test = np.squeeze(np.concatenate(y_test, 0))

    if args.mode == 'full':
        full(args,x_test,y_test)
    elif args.mode == 'final':
        final(args,x_test,y_test)
    elif args.mode == 'draw':
        draw(args,x_test,y_test)
    elif args.mode == 'tsne':
        tsne(args, x_test, y_test)