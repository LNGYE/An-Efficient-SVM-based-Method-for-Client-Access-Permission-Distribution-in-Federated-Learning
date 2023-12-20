import copy
import math
import random
import statistics
import struct
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from getData import GetDataSet
from models import (Cifar10_CNN, Cifar10_CNN3, CifarCnn, Mnist_2NN, Mnist_CNN,
                    ResNet18)


def weights_init_normal(m):
    """
    初始化模型的权重和偏差
    Args:
        m: A torch model to initialize.
    Returns:
        None.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
        torch.nn.init.constant_(m.bias.data, 0)


class client(object):
    def __init__(self, args, trainDataSet, testDataSet, dev):
        self.train_ds = trainDataSet
        self.test_ds = testDataSet
        self.dev = dev
        self.args = args
        self.train_dl = None 
        self.test_dl = DataLoader(self.test_ds, batch_size=128, shuffle=True)
        self.local_parameters = None

    def localUpdate(self, Net, client, lossFun, opti, global_parameters, isolated=False):
        if self.local_parameters == None:
            self.local_parameters = global_parameters
        # 首先加载全局模型
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=self.args.batchsize, shuffle=True)
        if isolated:
            self.args.epoch = 200
        for epoch in range(self.args.epoch):
            sum_loss = 0
            cnt = 0
            sum_accu = 0
            batch_loss = []
            # 这里的测试集和训练集用的是每个客户端自己的
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                preds_idx = torch.argmax(preds, dim=1)
                sum_accu += (preds_idx == label).float().mean()
                loss.backward()
                opti.step()
                opti.zero_grad()
                sum_loss += loss.item()
                cnt += 1
            # print("epoch {} loss {} accuracy {}".format(epoch, sum_loss/cnt, sum_accu/cnt))
            local_accuracy = sum_accu/cnt

        client = client
        list = [client, local_accuracy.item()]
        data = pd.DataFrame([list])
        # data.to_csv('fl-svm-1/实验测试结果/' + 'client_local_accuracy_{}.csv'.format(self.args.model_name),mode='a', header=False, index=False)

        self.local_parameters = copy.deepcopy(Net.state_dict())
        return Net.state_dict()   # 返回更新后的本地模型


class ClientsGroup(object):
    def __init__(self, args, datasetname, dev):
        self.args = args
        self.data_set_name = datasetname
        self.num_of_clients = args.num_of_clients
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None

        if 'femnist' in self.args.model_name:
            self.dataSetBalanceAllocation_femnist()
        elif 'shakespeare' in self.args.model_name:
            self.dataSetBalanceAllocation_shakespeare()
        else:
            self.dataSetBalanceAllocation()

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

            if np.random.random() < 0.5:
                new_image = cv2.flip(new_image, 1)

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(data.shape[1] * data.shape[2] * data.shape[3])])
            new_image = (new_image - mean) / std

            new_images.append(new_image)

        return new_images

    def dataSetBalanceAllocation(self):
        DataSet = GetDataSet(self.data_set_name,
                             self.args.partition)

        test_data = torch.tensor(DataSet.test_data)
        test_label = torch.argmax(torch.tensor(DataSet.test_label), dim=1)

        # print("test_data:", test_data)
        # print("test_label:", test_label)
        self.test_data_loader = DataLoader(TensorDataset(
            test_data, test_label), batch_size=1024, shuffle=False)
        preprocess = 1 if self.data_set_name == 'cifar10' else 0  # 0

        train_data = DataSet.train_data
        train_label = DataSet.train_label

        test_data_client = DataSet.test_data_client
        test_label_client = DataSet.test_label_client

        # partition:str="noniid-labeldir"
        if self.args.partition in ('homo', 'noniid-#label2'):
            shard_size = DataSet.train_data_size // self.num_of_clients // 2
            shards_id = np.random.permutation(
                DataSet.train_data_size // shard_size)

            shard_size_test = DataSet.test_data_size // self.num_of_clients // 2
            # np.random.permutation(mnistDataSet.test_data_size // shard_size_test)
            shards_id_test = shards_id
            for i in range(self.num_of_clients):
                shards_id1 = shards_id[i * 2]
                shards_id2 = shards_id[i * 2 + 1]

                data_shards1 = train_data[shards_id1 *
                                          shard_size: shards_id1 * shard_size + shard_size]
                data_shards2 = train_data[shards_id2 *
                                          shard_size: shards_id2 * shard_size + shard_size]
                label_shards1 = train_label[shards_id1 *
                                            shard_size: shards_id1 * shard_size + shard_size]
                label_shards2 = train_label[shards_id2 *
                                            shard_size: shards_id2 * shard_size + shard_size]
                if preprocess:
                    local_data, local_label = self.preprocess(np.vstack(
                        (data_shards1, data_shards2))), np.vstack((label_shards1, label_shards2))

                else:
                    local_data, local_label = np.vstack(
                        (data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))

                # for i in range(self.num_of_clients):
                shards_id1_test = shards_id_test[i * 2]
                shards_id2_test = shards_id_test[i * 2 + 1]

                data_shards1_test = test_data_client[shards_id1_test *
                                                     shard_size_test: shards_id1_test * shard_size_test + shard_size_test]
                data_shards2_test = test_data_client[shards_id2_test *
                                                     shard_size_test: shards_id2_test * shard_size_test + shard_size_test]
                label_shards1_test = test_label_client[shards_id1_test *
                                                       shard_size_test: shards_id1_test * shard_size_test + shard_size_test]
                label_shards2_test = test_label_client[shards_id2_test *
                                                       shard_size_test: shards_id2_test * shard_size_test + shard_size_test]

                local_data_test, local_label_test = np.vstack(
                    (data_shards1_test, data_shards2_test)), np.vstack((label_shards1_test, label_shards2_test))

                local_label = np.argmax(local_label, axis=1)
                local_label_test = np.argmax(local_label_test, axis=1)
                someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data)), torch.tensor(np.array(local_label))), TensorDataset(
                    torch.tensor(np.array(local_data_test)), torch.tensor(np.array(local_label_test))), self.dev)
                self.clients_set['client{}'.format(i)] = someone

        elif self.args.partition == "noniid-#label1" or (self.args.partition > "noniid-#label3" and self.args.partition <= "noniid-#label9"):

            train_label_dense = np.argmax(train_label, axis=1)
            test_label_client_dense = np.argmax(test_label_client, axis=1)
            num = eval(self.args.partition[13:])
            K = 10
            times = [0 for i in range(10)]
            contain = []
            for i in range(self.args.num_of_clients):
                current = [i % K]
                times[i % K] += 1
                j = 1
                while (j < num):
                    ind = random.randint(0, K-1)
                    if (ind not in current):
                        j = j+1
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)
            net_dataidx_map = {i: np.ndarray(
                0, dtype=np.int64) for i in range(self.args.num_of_clients)}
            for i in range(K):
                idx_k = np.where(train_label_dense == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(self.args.num_of_clients):
                    if i in contain[j]:
                        net_dataidx_map[j] = np.append(
                            net_dataidx_map[j], split[ids])
                        ids += 1

            net_dataidx_map_test = {i: np.ndarray(
                0, dtype=np.int64) for i in range(self.args.num_of_clients)}
            for i in range(K):
                idx_k = np.where(test_label_client_dense == i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k, times[i])
                ids = 0
                for j in range(self.args.num_of_clients):
                    if i in contain[j]:
                        net_dataidx_map_test[j] = np.append(
                            net_dataidx_map_test[j], split[ids])
                        ids += 1
            for i in range(self.args.num_of_clients):

                if preprocess:
                    local_data, local_label = self.preprocess(
                        train_data[net_dataidx_map[i]]), train_label_dense[net_dataidx_map[i]]

                else:
                    local_data, local_label = train_data[net_dataidx_map[i]
                                                         ], train_label_dense[net_dataidx_map[i]]
                local_data_test, local_label_test = test_data_client[net_dataidx_map_test[i]
                                                                     ], test_label_client_dense[net_dataidx_map_test[i]]

                someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data)), torch.tensor(np.array(local_label))), TensorDataset(
                    torch.tensor(np.array(local_data_test)), torch.tensor(np.array(local_label_test))), self.dev)
                self.clients_set['client{}'.format(i)] = someone

        elif self.args.partition == "noniid-labeldir":
            train_label_dense = np.argmax(train_label, axis=1)
            test_label_client_dense = np.argmax(test_label_client, axis=1)
            min_size = 0
            min_require_size = 10
            K = 10  
            N = train_data.shape[0]
            # np.random.seed(2020)
            net_dataidx_map = {}
            net_dataidx_map_test = {}

            while min_size < min_require_size:                
                idx_batch = [[] for _ in range(self.args.num_of_clients)]
                idx_batch_test = [[] for _ in range(self.args.num_of_clients)]

                """
                这段代码涉及到数据的分配和划分过程，主要用于将训练集和测试集中的样本按类别进行划分，并分配给不同的客户端
                具体来说,代码中的循环是对K个类别(假设类别编号从0到K-1)进行遍历,然后按照随机比例将每个类别的样本分配给不同的客户端
                """
                for k in range(K):
                    idx_k = np.where(train_label_dense == k)[0]
                    np.random.shuffle(idx_k)

                    idx_k_test = np.where(test_label_client_dense == k)[0]
                    np.random.shuffle(idx_k_test)

                    proportions = np.random.dirichlet(
                        np.repeat(self.args.beta, self.args.num_of_clients))

                    proportions = np.array(
                        [p * (len(idx_j) < N / self.args.num_of_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions_train = (
                        np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,
                                 idx in zip(idx_batch, np.split(idx_k, proportions_train))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    proportions_test = (
                        np.cumsum(proportions) * len(idx_k_test)).astype(int)[:-1]
                    idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(
                        idx_batch_test, np.split(idx_k_test, proportions_test))]

            """
            这段代码用于为每个客户端创建一个客户端对象，并将训练集和测试集的数据分配给各个客户端
            """
            for i in range(self.args.num_of_clients):
                np.random.shuffle(idx_batch[i])
                net_dataidx_map[i] = idx_batch[i]
                np.random.shuffle(idx_batch_test[i])
                net_dataidx_map_test[i] = idx_batch_test[i]
                if preprocess:
                    local_data, local_label = self.preprocess(
                        train_data[net_dataidx_map[i]]), train_label_dense[net_dataidx_map[i]]

                else:
                    local_data, local_label = train_data[net_dataidx_map[i]
                                                         ], train_label_dense[net_dataidx_map[i]]
                local_data_test, local_label_test = test_data_client[net_dataidx_map_test[i]
                                                                     ], test_label_client_dense[net_dataidx_map_test[i]]

                someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data)), torch.tensor(np.array(local_label))), TensorDataset(
                    torch.tensor(np.array(local_data_test)), torch.tensor(np.array(local_label_test))), self.dev)
                self.clients_set['client{}'.format(i)] = someone

        elif self.args.partition == "iid-diff-quantity":
            train_label_dense = np.argmax(train_label, axis=1)
            test_label_client_dense = np.argmax(test_label_client, axis=1)

            idxs = np.random.permutation(len(train_data))
            idxs_test = np.random.permutation(len(test_data_client))
            min_size = 0
            while min_size < 10:
                proportions = np.random.dirichlet(
                    np.repeat(self.args.beta, self.args.num_of_clients))
                proportions = proportions/proportions.sum()
                min_size = np.min(proportions*len(idxs))

            proportions_train = (np.cumsum(proportions) *
                                 len(idxs)).astype(int)[:-1]
            batch_idxs = np.split(idxs, proportions_train)
            net_dataidx_map = {i: batch_idxs[i]
                               for i in range(self.args.num_of_clients)}

            proportions_test = (np.cumsum(proportions) *
                                len(idxs_test)).astype(int)[:-1]
            batch_idxs_test = np.split(idxs_test, proportions_test)
            net_dataidx_map_test = {
                i: batch_idxs_test[i] for i in range(self.args.num_of_clients)}

            for i in range(self.args.num_of_clients):

                if preprocess:
                    local_data, local_label = self.preprocess(
                        train_data[net_dataidx_map[i]]), train_label_dense[net_dataidx_map[i]]

                else:
                    local_data, local_label = train_data[net_dataidx_map[i]
                                                         ], train_label_dense[net_dataidx_map[i]]
                local_data_test, local_label_test = test_data_client[net_dataidx_map_test[i]
                                                                     ], test_label_client_dense[net_dataidx_map_test[i]]

                someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data)), torch.tensor(np.array(local_label))), TensorDataset(
                    torch.tensor(np.array(local_data_test)), torch.tensor(np.array(local_label_test))), self.dev)
                self.clients_set['client{}'.format(i)] = someone

    def dataSetBalanceAllocation_femnist(self):
        DataSet = GetDataSet(self.data_set_name, self.args.partition)
        test_data = DataSet.test_data
        test_label = DataSet.test_label
        self.test_data_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(
            test_label, dtype=torch.int64)), batch_size=1024, shuffle=False)

        partitioned_train_data = DataSet.partitioned_train_data
        partitioned_test_data = DataSet.partitioned_test_data

        for i in range(self.num_of_clients):
            key = DataSet.sampled_100[i]
            local_data, local_label = partitioned_train_data[key]['x'], partitioned_train_data[key]['y']
            local_data_test, local_label_test = partitioned_test_data[
                key]['x'], partitioned_test_data[key]['y']

            someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data), dtype=torch.float32), torch.tensor(np.array(local_label), dtype=torch.int64)), TensorDataset(
                torch.tensor(np.array(local_data_test), dtype=torch.float32), torch.tensor(np.array(local_label_test), dtype=torch.int64)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

    def dataSetBalanceAllocation_shakespeare(self):
        DataSet = GetDataSet(self.data_set_name, self.args.partition)
        test_data = DataSet.test_data
        test_label = DataSet.test_label
        self.test_data_loader = DataLoader(TensorDataset(torch.tensor(test_data, dtype=torch.long).to(
            self.dev), torch.tensor(test_label, dtype=torch.long).to(self.dev)), batch_size=1024, shuffle=False)

        partitioned_train_data = DataSet.partitioned_train_data
        partitioned_test_data = DataSet.partitioned_test_data
        for i in range(self.num_of_clients):
            key = DataSet.sampled_100[i]
            local_data, local_label = partitioned_train_data[key]['x'], partitioned_train_data[key]['y']
            local_data_test, local_label_test = partitioned_test_data[
                key]['x'], partitioned_test_data[key]['y']

            someone = client(self.args, TensorDataset(torch.tensor(np.array(local_data), dtype=torch.long).to(self.dev), torch.tensor(np.array(local_label), dtype=torch.long).to(
                self.dev)), TensorDataset(torch.tensor(np.array(local_data_test), dtype=torch.long).to(self.dev), torch.tensor(np.array(local_label_test), dtype=torch.long).to(self.dev)), self.dev)
            self.clients_set['client{}'.format(i)] = someone
