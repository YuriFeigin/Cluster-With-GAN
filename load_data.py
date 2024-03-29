import os
import tensorflow as tf
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.cluster import KMeans

STL_PATH = '/root/Documents/Cluster-With-GAN/datasets/stl10_binary'
STL_DATA_PATH_TRAIN = os.path.join(STL_PATH, 'train_X.bin')
STL_LABEL_PATH_TRAIN = os.path.join(STL_PATH, 'train_y.bin')
STL_DATA_PATH_TEST = os.path.join(STL_PATH, 'test_X.bin')
STL_LABEL_PATH_TEST = os.path.join(STL_PATH, 'test_y.bin')
STL_DATA_PATH_UNLABELD = os.path.join(STL_PATH, 'unlabeled_X.bin')

CELEBA_PATH = '/dl_data/datasets/CelebA/'


def Load(name, data, shuffle, batch_size, pad_size, img_size):
    with tf.device("/cpu:0"):
        if name == "GMM":
            dataset = GMM.DataSet(data, shuffle, batch_size)
        elif name == "mnist":
            dataset = mnist.DataSet(data, shuffle, batch_size)
        elif name == "cifar10":
            dataset = cifar10(data, shuffle, batch_size, pad_size, img_size)
        elif name == "cifar100":
            dataset = cifar100(data, shuffle, batch_size, pad_size, img_size)
        elif name == "stl10":
            dataset = stl10(data, shuffle, batch_size, pad_size, img_size)
        elif name == "imnet32":
            if data == 'train':
                data_path = '/dl_data/users/Yuri/Small_ImageNet/train_32x32/*.tfrecords'
            elif data == 'valid':
                data_path = '/dl_data/users/Yuri/Small_ImageNet/valid_32x32/*.tfrecords'
            x = imagenet32.load(data_path, mode, batch_size)
        elif name == "celeba":
            dataset = celeba(data, shuffle, batch_size, img_size)
        else:
            raise ValueError("Unknown dataset.")
    return dataset



class DataSet:
    def __init__(self):
        self.is_init = False
        self.is_imgs = None
        self.resize = None

    def build_dataset(self, shuffle, batch_size, img_size, map_func=None):
        def dataset_map_resize(img, labels):
            img = tf.image.resize_images(img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
            # img = tf.clip_by_value(img, 0, 255)
            return img, labels

        def random_crop(img, labels):
            # shape = tf.shape(img)
            # img = tf.pad(img, [[4] * 2, [4] * 2, [0] * 2], mode='REFLECT')
            img = tf.random_crop(img, [img_size, img_size, self.imgs.shape[-1]])
            return img, labels

        if self.is_imgs:
            self.imgs_placeholder = tf.placeholder(self.imgs.dtype, [None, img_size + 2 * self.pad_size, img_size + 2 * self.pad_size, self.imgs.shape[-1]])
        else:
            self.imgs_placeholder = tf.placeholder(self.imgs.dtype, [None, ])
        self.labels_placeholder = tf.placeholder(self.labels.dtype, [None] + list(self.labels.shape[1:]))
        dataset = tf.data.Dataset.from_tensor_slices((self.imgs_placeholder, self.labels_placeholder))
        if map_func is not None:
            dataset = dataset.map(map_func, num_parallel_calls=16)
        if self.resize and img_size > 100 and not self.is_imgs:
            dataset = dataset.map(dataset_map_resize, num_parallel_calls=16)
        dataset = dataset.map(random_crop, num_parallel_calls=4)
        if shuffle:
            dataset = dataset.shuffle(4096)
            dataset = dataset.batch(batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat(1).prefetch(8)
        self.iterator = dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

    def set_clustring(self, path, n_clusters,n_cluster_hist):
        print('start calculate clustering')
        np_data = []
        files = np.array(os.listdir(os.path.join(path, 'latent')))
        IterNumExist = np.array([int(f[6:-4]) for f in files])
        ind = np.argsort(IterNumExist)[-n_cluster_hist:]
        for i in ind:
            file = os.path.join(path,'latent', files[i])
            np_data.append(np.load(file)['latent'])
        X = np.concatenate(np_data, 1)
        kmeans = KMeans(n_clusters=n_clusters, precompute_distances=True, n_jobs=32)
        kmeans.fit(X)
        self.labels = kmeans.labels_
        _, self.label_dist = np.unique(self.labels, return_counts=True)
        self.label_dist = self.label_dist / self.label_dist.sum()
        print('end calculate clustering')

    def set_sub_clustring(self, path, n_clusters):
        print('start calculate clustering')
        np_data = []
        files = np.array(os.listdir(os.path.join(path, 'latent')))
        IterNumExist = np.array([int(f[6:-4]) for f in files])
        ind = np.argsort(IterNumExist)[-10:]
        for i in ind:
            file = os.path.join(path,'latent', files[i])
            np_data.append(np.load(file)['latent'])
        X = np.concatenate(np_data, 1)

        y_pred = self.labels.copy()
        kmeans = KMeans(n_clusters=n_clusters, precompute_distances=True, n_jobs=32)
        for i, l in enumerate(np.unique(self.labels)):
            t_X = X[self.labels == l, :]
            kmeans.fit(t_X)
            y_pred[self.labels == l] = kmeans.labels_ + i * n_clusters
        self.labels = y_pred
        _, self.label_dist = np.unique(self.labels, return_counts=True)
        self.label_dist = self.label_dist / self.label_dist.sum()
        print('end calculate clustering')

    def load_sub_imgs(self,sz):
        if self.is_init:
            images = np.random.permutation(self.imgs)[:sz]
        else:
            assert 'dataset not initial'
        return images

    def get_full_next(self):
        return self.next_element

    def get_imgs_next(self):
        return self.next_element[0]

    def get_label_dist(self):
        return self.label_dist

    def _init_dataset(self):
        ind = self.ind.get(self.data)
        self.imgs = self.imgs[ind]
        self.labels = self.labels[ind]
        if (self.resize and self.img_size <= 100 and self.is_imgs) or self.pad_size:
            print('start resize', flush=True)
            self.imgs = np.stack([resize(img, (self.img_size+2*self.pad_size, self.img_size + 2*self.pad_size), anti_aliasing=True) * 255 for img in self.imgs], 0)
            print('finish resize', flush=True)
        self.is_init = True
        
    def init_dataset(self, sess):
        if not self.is_init:
            self._init_dataset()
        if self.shuffle:
            ind = np.random.permutation(self.imgs.shape[0])
        else:
            ind = np.ones(self.imgs.shape[0],np.bool)
        sess.run(self.iterator.initializer, feed_dict={self.imgs_placeholder: self.imgs[ind],
                                                       self.labels_placeholder: self.labels[ind]})

class cifar10(DataSet):
    def __init__(self, data, shuffle, batch_size, pad_size, img_size):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_size = pad_size
        if img_size is None or img_size == 32:
            self.img_size = 32
            self.resize = False
        else:
            self.img_size = img_size
            self.resize = True
        self.is_imgs = True
        self.num_classes = 10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        self.imgs = np.concatenate([x_train, x_test], 0)
        self.labels = np.concatenate([y_train, y_test], 0)
        self.label_dist = np.ones(self.num_classes) / self.num_classes

        train_ind = np.concatenate([np.ones_like(y_train, np.bool), np.zeros_like(y_test, np.bool)], 0)
        test_ind = np.concatenate([np.zeros_like(y_train, np.bool), np.ones_like(y_test, np.bool)], 0)

        self.ind = {}
        self.ind['train'] = train_ind
        self.ind['test'] = test_ind
        self.ind['all'] = np.logical_or(train_ind, test_ind)
        self.build_dataset(shuffle, batch_size, self.img_size)

class cifar100(DataSet):
    def __init__(self, data, shuffle, batch_size, pad_size, img_size):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_size = pad_size
        if img_size is None or img_size == 32:
            self.img_size = 32
            self.resize = False
        else:
            self.img_size = img_size
            self.resize = True
        self.is_imgs = True
        self.num_classes = 20
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data("coarse")
        y_train = np.squeeze(y_train)
        y_test = np.squeeze(y_test)
        self.imgs = np.concatenate([x_train, x_test], 0)
        self.labels = np.concatenate([y_train, y_test], 0)
        self.label_dist = np.ones(self.num_classes) / self.num_classes

        train_ind = np.concatenate([np.ones_like(y_train, np.bool), np.zeros_like(y_test, np.bool)], 0)
        test_ind = np.concatenate([np.zeros_like(y_train, np.bool), np.ones_like(y_test, np.bool)], 0)

        self.ind = {}
        self.ind['train'] = train_ind
        self.ind['test'] = test_ind
        self.ind['all'] = np.logical_or(train_ind,test_ind)
        self.build_dataset(shuffle, batch_size, self.img_size)
        
class stl10(DataSet):
    def __init__(self, data, shuffle, batch_size, pad_size, img_size):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.pad_size = pad_size
        if img_size is None or img_size == 96:
            self.img_size = 96
            self.resize = False
        else:
            self.img_size = img_size
            self.resize = True
        self.is_imgs = True
        self.num_classes = 10

        x_train = self.read_images(STL_DATA_PATH_TRAIN)
        y_train = self.read_labels(STL_LABEL_PATH_TRAIN)
        x_test = self.read_images(STL_DATA_PATH_TEST)
        y_test = self.read_labels(STL_LABEL_PATH_TEST)
        x_unlabeled = self.read_images(STL_DATA_PATH_UNLABELD)
        y_unlabeled = np.ones(x_unlabeled.shape[0], dtype='int32') * -1
        self.imgs = np.concatenate([x_train, x_test, x_unlabeled], 0)
        self.labels = np.concatenate([y_train, y_test, y_unlabeled], 0)
        self.label_dist = np.ones(self.num_classes) / self.num_classes

        train_ind = np.concatenate([np.ones_like(y_train, np.bool), np.zeros_like(y_test, np.bool),
                                    np.zeros_like(y_unlabeled, np.bool)], 0)
        test_ind = np.concatenate([np.zeros_like(y_train, np.bool), np.ones_like(y_test, np.bool),
                                    np.zeros_like(y_unlabeled, np.bool)], 0)
        unlabeled_ind = np.concatenate([np.zeros_like(y_train, np.bool), np.zeros_like(y_test, np.bool),
                                    np.ones_like(y_unlabeled, np.bool)], 0)

        self.ind = {}
        self.ind['train'] = np.logical_or(train_ind, unlabeled_ind)
        self.ind['train_label'] = train_ind
        self.ind['test'] = test_ind
        self.ind['all'] = np.logical_or(self.ind['train'], test_ind)

        self.build_dataset(shuffle, batch_size, self.img_size)
        
    @staticmethod
    def read_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 3, 2, 1))
            return images
        
    @staticmethod
    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8) - 1
            return labels


class celeba(DataSet):
    def __init__(self, data, shuffle, batch_size, img_size):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        if img_size is None or img_size == 64:
            self.img_size = 64
            self.resize = False
        else:
            self.img_size = img_size
            self.resize = True
        self.is_imgs = False
        self.num_classes = None

        self.partition_fn = partition_fn = os.path.join(CELEBA_PATH, 'list_eval_partition.txt')
        with open(partition_fn, "r") as infile:
            img_fn_list = infile.readlines()
        self.img_fn_list = img_fn_list
        imgs = []
        imgs_datasets = []
        for elems in img_fn_list:
            elem = elems.strip().split()
            imgs.append(os.path.join(CELEBA_PATH, 'img_align_celeba', elem[0]))
            imgs_datasets.append(int(elem[1]))
        self.imgs = np.array(imgs)
        imgs_datasets = np.array(imgs_datasets)
        self.labels = np.ones(self.imgs.shape[0], dtype='int32') * -1
        self.label_dist = None

        self.ind = {}
        self.ind['train'] = imgs_datasets == 0
        self.ind['valid'] = imgs_datasets == 1
        self.ind['test'] = imgs_datasets == 2
        self.ind['all'] = np.ones(self.imgs.shape[0], np.bool)
        def map_func_1(path):
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img, 3)
            crop_size = 108
            re_size = 64
            img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size,
                                                crop_size)
            img = tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            return img

        def map_func_2(path,labels):
            img = tf.read_file(path)
            img = tf.image.decode_jpeg(img, 3)
            img = img[40:188, 15:163, :]
            img = tf.image.resize_images(img, [self.img_size, self.img_size], method=tf.image.ResizeMethod.BILINEAR)
            # img = tf.image.resize_images(img, [img_size, img_size], method=0, align_corners=False)
            # img = (tf.cast(img, tf.float32))  # / 256. # + tf.random_uniform(tf.shape(img))
            # if mode == 'train':
            #     img = tf.image.random_flip_left_right(img)
            return img, labels

        self.build_dataset(shuffle, batch_size, self.img_size, map_func_2)

    def load_sub_imgs(self,sz):
        if self.is_init:
            imgs = np.random.permutation(self.imgs)[:sz]
            images = np.stack([resize(io.imread(f)[40:188, 15:163, :], (self.img_size, self.img_size), anti_aliasing=True)*255 for f in imgs])
            return images

