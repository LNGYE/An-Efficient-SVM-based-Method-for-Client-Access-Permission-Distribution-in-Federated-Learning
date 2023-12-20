import gzip
import json
import os
import pickle
import platform
import struct
from collections import defaultdict

import numpy as np
from keras.utils.np_utils import to_categorical
from torchvision import datasets, transforms

from utils import letter_to_vec, word_to_indices


def load_data(mode):
    # os.system('pwd')
    if mode == 'train':
        file_path = 'data/MNIST/train-images-idx3-ubyte'
        label_path = 'data/MNIST/train-labels-idx1-ubyte'
    else:
        file_path = 'data/MNIST/t10k-images-idx3-ubyte'
        label_path = 'data/MNIST/t10k-labels-idx1-ubyte'

    binfile = open(file_path, 'rb')
    buffers = binfile.read()
    magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from(
        '>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])

    images = images.reshape((len(images), 28, 28, 1))

    binfile = open(label_path, 'rb')
    buffers = binfile.read()
    magic, num = struct.unpack_from('>II', buffers, 0)
    labels = struct.unpack_from(
        '>' + str(num) + "B", buffers, struct.calcsize('>II'))
    binfile.close()
    labels = np.reshape(labels, [num])

    images = images/255
    labels = to_categorical(labels)

    return images, labels


class GetDataSet(object):
    def __init__(self, dataSetName, partition):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None
        self._index_in_train_epoch = 0

        if self.name == 'mnist' or self.name == 'fmnist':
            self.mnistDataSetConstruct(partition)
        elif self.name == 'femnist':
            self.femnistDataSetConstruct(partition)
        elif self.name == 'cifar10':
            self.cifar10_dataset_construct(partition)
        elif self.name == 'shakespeare':
            self.ShakeSpeareDataSetConstruct(partition)
        else:
            pass

    def ShakeSpeareDataSetConstruct(self, partition):
        train_clients, train_groups, train_data_temp, test_data_temp = read_data(r"data/SHAKESPEARE/train",
                                                                                 r"data/SHAKESPEARE/test")
        all_keys = train_clients
        # print(len(train_clients)) # 660
        self.sampled_500 = all_keys[:500]  # NOTE: 100 removed
        train_data_temp_tmp = {
            key: train_data_temp[key] for key in self.sampled_500}
        test_data_temp_tmp = {
            key: test_data_temp[key] for key in self.sampled_500}
        train_data_temp = {}
        test_data_temp = {}

        for i in self.sampled_500:
            cur_x = train_data_temp_tmp[i]['x']
            cur_y = train_data_temp_tmp[i]['y']
            idx_x = [word_to_indices(cur_x[0])]
            idx_y = [letter_to_vec(cur_y[0])]
            for j in range(1, len(cur_x)):
                # train_data_temp[i]= {'x':word_to_indices(cur_x[j]),'y':letter_to_vec(cur_y[j])}
                idx_x.append(word_to_indices(cur_x[j]))
                idx_y.append(letter_to_vec(cur_y[j]))
            train_data_temp[i] = {'x': idx_x, 'y': idx_y}

        for i in self.sampled_500:
            cur_x = test_data_temp_tmp[i]['x']
            cur_y = test_data_temp_tmp[i]['y']
            idx_x = [word_to_indices(cur_x[0])]
            idx_y = [letter_to_vec(cur_y[0])]
            for j in range(len(cur_x)):
                # test_data_temp[i]= {'x':word_to_indices(cur_x[j]),'y':letter_to_vec(cur_y[j])}
                idx_x.append(word_to_indices(cur_x[j]))
                idx_y.append(letter_to_vec(cur_y[j]))
            test_data_temp[i] = {'x': idx_x, 'y': idx_y}
        test_data_x = []
        test_data_y = []

        for i in test_data_temp.keys():
            cur_x = test_data_temp[i]['x']
            cur_y = test_data_temp[i]['y']
            for j in range(len(cur_x)):
                test_data_x.append(cur_x[j])
                test_data_y.append(cur_y[j])

        self.test_data = test_data_x
        self.test_label = test_data_y

        self.partitioned_train_data = train_data_temp
        self.partitioned_test_data = test_data_temp

        # def __getitem__(self, index):
        #     sentence, target = self.data[index], self.label[index]
        #     indices = word_to_indices(sentence)
        #     target = letter_to_vec(target)
        #     # y = indices[1:].append(target)
        #     # target = indices[1:].append(target)
        #     indices = torch.LongTensor(np.array(indices))
        #     # y = torch.Tensor(np.array(y))
        #     # target = torch.LongTensor(np.array(target))
        #     return indices, target

    def mnistDataSetConstruct(self, partition):
        if self.name == 'mnist':
            data_dir = r'data/MNIST'
        else:
            data_dir = r'data/FMNIST'

        train_images_path = os.path.join(
            data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(
            data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)
        # train_images,train_labels=load_data('train')
        # test_images,test_labels=load_data('test')

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]


        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1

        
        train_images = train_images.reshape(
            train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(
            test_images.shape[0], test_images.shape[1] * test_images.shape[2])


        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if 'noniid-#label2' == partition:

            labels = np.argmax(train_labels, axis=1)

            order = np.argsort(labels)

            self.train_data = train_images[order]
            self.train_label = train_labels[order]

            labels_test = np.argmax(test_labels, axis=1)
            order_test = np.argsort(labels_test)
            self.test_data_client = test_images[order_test]
            self.test_label_client = test_labels[order_test]
        else:  # partition:str="noniid-labeldir"
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)

            self.train_data = train_images[order]
            self.train_label = train_labels[order]

            labels_test = np.argmax(test_labels, axis=1)
            order_test = np.argsort(labels_test)

            np.random.shuffle(order_test)
            self.test_data_client = test_images[order_test]
            self.test_label_client = test_labels[order_test]
        # 这里的test数据集相当于始终没有变
        self.test_data = test_images
        self.test_label = test_labels

    def femnistDataSetConstruct(self, partition):

        train_clients, train_groups, train_data_temp, test_data_temp = read_data(
            "data/FEMNIST/train", "data/FEMNIST/test")
        all_keys = train_clients
        self.sampled_100 = all_keys[:100] 
        train_data_temp = {key: train_data_temp[key]
                           for key in self.sampled_100}

        test_data_temp = {key: test_data_temp[key] for key in self.sampled_100}
        self.test_data = np.array(test_data_temp[self.sampled_100[0]]['x'])
        self.test_label = np.array(test_data_temp[self.sampled_100[0]]['y'])

        for i in range(1, len(self.sampled_100)):
            self.test_data = np.concatenate((self.test_data, np.array(
                test_data_temp[self.sampled_100[i]]['x'])), axis=0)
            self.test_label = np.concatenate((self.test_label, np.array(
                test_data_temp[self.sampled_100[i]]['y'])), axis=0)

        self.partitioned_train_data = train_data_temp
        self.partitioned_test_data = test_data_temp

    def preprocess(self, data):
        new_images = []
        shape = (24, 24, 3)
        for i in range(data.shape[0]):
            old_image = data[i, :, :, :]
            old_image = np.transpose(old_image, (1, 2, 0))

            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left +
                                  shape[0], top: top + shape[1], :]

            # if np.random.random() < 0.5:
            #     new_image = cv2.flip(new_image, 1)
            # #new_image=old_image

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(data.shape[1] * data.shape[2] * data.shape[3])])
            new_image = (new_image - mean) / std
            new_images.append(new_image)

        return np.array(new_images)

    def cifar10_dataset_construct(self, partition):
        images, labels = [], []

        with open(r'data/CIFAR10/cifar-10-batches-py/test_batch', 'rb') as fo:
            if 'Windows' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')
            elif 'Linux' in platform.platform():
                cifar10 = pickle.load(fo, encoding='bytes')

        for i in range(len(cifar10[b'labels'])):
            image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
            image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.float32)
            images.append(image)
        labels += cifar10[b'labels']
        images = np.array(images)
        labels = np.array(labels, dtype='int')
        self.test_label = dense_to_one_hot(labels)
        self.test_data = []

        shape = (24, 24, 3)
        for i in range(images.shape[0]):
            old_image = images[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            old_image = old_image[left: left +
                                  shape[0], top: top + shape[1], :]

            mean = np.mean(old_image)
            std = np.max([np.std(old_image),
                          1.0 / np.sqrt(images.shape[1] * images.shape[2] * images.shape[3])])
            new_image = (old_image - mean) / std
            self.test_data.append(new_image)
        self.test_data = np.array(self.test_data)

        images, labels = [], []

        for filename in [r'data/CIFAR10/cifar-10-batches-py/test_batch_{}'.format(i) for i in range(1, 6)]:
            with open(filename, 'rb') as fo:
                if 'Windows' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')
                elif 'Linux' in platform.platform():
                    cifar10 = pickle.load(fo, encoding='bytes')

            for i in range(len(cifar10[b'labels'])):
                image = np.reshape(cifar10[b'data'][i], (3, 32, 32))
                # image = np.transpose(image, (1, 2, 0))
                image = image.astype(np.float32)
                images.append(image)
            labels += cifar10[b'labels']
        images = np.array(images)
        labels = np.array(labels, dtype='int')

        # self.train_data, self.train_label = images, labels
        if 'noniid-#label2' == partition:
            order = np.arange(images.shape[0])
            np.random.shuffle(order)
            self.train_data = images[order]
            self.train_label = dense_to_one_hot(labels[order])

            labels_test = np.argmax(self.test_label, axis=1)
            order_test = np.argsort(labels_test)
            np.random.shuffle(order_test)
            self.test_data_client = self.test_data[order_test]
            self.test_label_client = self.test_label[order_test]
        else:
            order = np.argsort(labels)
            self.train_data = images[order]
            self.train_label = dense_to_one_hot(labels[order])
            labels_test = np.argmax(self.test_label, axis=1)
            order_test = np.argsort(labels_test)
            self.test_data_client = self.test_data[order_test]
            self.test_label_client = self.test_label[order_test]

        self.train_data_size = self.train_data.shape[0]
        self.test_data_size = self.test_data.shape[0]


def next_batch(self, batch_size):
    start = self._index_in_train_epoch
    self._index_in_train_epoch += batch_size
    if self._index_in_train_epoch > self.train_data_size:
        order = np.arange(self.train_data_size)
        np.random.shuffle(order)
        self.train_data = self.train_data[order]
        self.train_label = self.train_label[order]

        start = 0
        self._index_in_train_epoch = batch_size
        assert batch_size <= self.train_data_size
    end = self._index_in_train_epoch
    return self.train_data[start: end], self.train_label[start: end]


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        # bytestream.close()
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


# if __name__ == "__main__":
#     'test data set'
#     mnistDataSet = GetDataSet('mnist', True)  # test NON-IID
#     if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
#             type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
#         print('the type of data is numpy ndarray')
#     else:
#         print('the type of data is not numpy ndarray')
#     print('the shape of the train data set is {}'.format(
#         mnistDataSet.train_data.shape))
#     print('the shape of the test data set is {}'.format(
#         mnistDataSet.test_data.shape))
#     print(mnistDataSet.train_label[0:100],
#           mnistDataSet.train_label[11000:11100])
