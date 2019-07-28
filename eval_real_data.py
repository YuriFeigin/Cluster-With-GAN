import os
import argparse
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import load_data
import utils.inception_score as inception_score
import utils.ndb as ndb



def main(args):
    # choose model
    dataset_train = load_data.Load(args.dataset, args.train_on, shuffle=False, batch_size=1000, img_size=args.img_size)
    next_element_train = dataset_train.get_imgs_next()
    dataset_test = load_data.Load(args.dataset, args.test_on, shuffle=False, batch_size=1000, img_size=args.img_size)
    next_element_test = dataset_test.get_imgs_next()
    init = tf.global_variables_initializer()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config) as sess:
        sess.run(init)
        # get images to calculate the fid mean, std and ndb
        dataset_train.init_dataset(sess)
        train_imgs = []
        while True:
            try:
                train_imgs.append(sess.run(next_element_train))
            except tf.errors.OutOfRangeError:
                break
        train_imgs = np.concatenate(train_imgs, 0)
        if args.eval == 'ndb':
            ndb_model = ndb.NDB(np.random.permutation(train_imgs)[:80000],max_dims=2000, semi_whitening=True)
        else:
            inception_score.update_fid_mean(train_imgs)

        # get images to eval inception scores and fid
        dataset_test.init_dataset(sess)
        test_imgs = []
        while True:
            try:
                test_imgs.append(sess.run(next_element_test))
            except tf.errors.OutOfRangeError:
                break
        test_imgs = np.concatenate(test_imgs, 0)
        if args.eval == 'is':
            inception_score_mean, inception_score_std, fid = inception_score.calc_scores(test_imgs)
            print('Inception score mean: ', inception_score_mean)
            print('Inception score std: ', inception_score_std)
        if args.eval == 'fid':
            inception_score_mean, inception_score_std, fid = inception_score.calc_scores(test_imgs)
            print('FID: ', fid)
        if args.eval == 'ndb':
            results_train = ndb_model.evaluate(train_imgs)
            results_test = ndb_model.evaluate(test_imgs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=['celeba', 'cifar10', 'cifar100', 'stl10'], type=str, help='choose dataset')
    parser.add_argument('eval', choices=['is', 'fid', 'ndb'], type=str, help='choose dataset')
    parser.add_argument('--train_on',default='train', choices=['train', 'test', 'labeled', 'all'], type=str,
                        help='on which images to train')
    parser.add_argument('--test_on',default='test', choices=['train', 'test', 'labeled', 'all'], type=str,
                        help='on which images to train')
    parser.add_argument('--img_size', default=32, type=int, help='the seed of the network initial')

    args = parser.parse_args()
    # save args to log file
    for arg, value in sorted(vars(args).items()):
        log_str = "Argument {}: {}".format(arg, value)
        print(log_str)
    main(args)
