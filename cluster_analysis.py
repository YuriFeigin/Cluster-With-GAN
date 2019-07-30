import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import logging
import random
import time
import numpy as np
import tensorflow as tf
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
            while CurIter < last_iter:  # for epoch
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

def get_clustering(args,hist_len,n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=True, n_jobs=32)
    files = np.array(os.listdir(os.path.join(args.data_path, 'latent')))
    IterNumExist = np.array([int(f[6:-4]) for f in files])
    argsort = np.argsort(IterNumExist)
    if hist_len > len(IterNumExist):
        return None
    cur_time_steps = IterNumExist[argsort[-hist_len:]]
    print('### running for history length of:', str(hist_len))
    print('using follow time steps:',cur_time_steps)
    data = []
    for i in cur_time_steps:
        data.append(np.load(os.path.join(args.data_path, 'latent', 'latent'+str(i)+'.npz'))['latent'])
    kmeans.fit(np.concatenate(data, 1))
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

def draw(args,x_test,y_test, n_clusters):
    shape = x_test.shape
    y_pred = get_clustering(args, 10, n_clusters)
    img = np.zeros([n_clusters * shape[1], 10 * shape[2], shape[3]])
    for i in range(n_clusters):
        ind = np.random.permutation(np.nonzero(y_pred == i)[0])[:10]
        img[shape[1] * i:shape[1] * (i + 1), :] = np.reshape(np.transpose(x_test[ind], [1, 0, 2, 3]),
                                                             [shape[1], shape[2] * 10, shape[3]]) / 255
    plt.imshow(img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='choose data')
    parser.add_argument('mode',choices=['full', 'final', 'draw'], type=str, help='choose dataset')
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
        draw(args,x_test,y_test,args.num_clusters)