import argparse
import copy
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from matplotlib.pylab import rcParams
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import optim
from tqdm import tqdm
from xgboost.sklearn import XGBClassifier

from cifar10_models import resnet20
from clients import ClientsGroup
from config import Config, parse_args
from models import (CharLSTM, Cifar10_CNN, Cifar10_CNN3, CifarCnn, FEMnist_CNN,
                    Mnist_2NN, Mnist_CNN, ResNet18, TransformerModel)
from train import train, train_isolated
from utils import (HfArgumentParser, dict_one, dict_slice, mkdir_if_needed,
                   scan_floder, set_seed)

PARALLEL = True
rcParams['figure.figsize'] = 12, 4

if __name__ == "__main__":
    if PARALLEL:
        args_add = parse_args()
    args = HfArgumentParser((Config)).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    set_seed(args.seed)

    if PARALLEL:
        args.loo_clients = args_add.loo_clients
        args.gpu = args_add.gpu
        args.client_save_path = args.save_path + args.method + '_' + args.model_name + '_' + args.partition
        args.start = args_add.start
        args.end = args_add.end

    # 创建目录
    print("args.save_path:", args.save_path)
    print("args.client_save_path:", args.client_save_path)
    mkdir_if_needed(args.save_path)
    mkdir_if_needed(args.client_save_path)

    # 用GPU训练
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    net_counter = None

    if args.model_name == 'mnist_2nn' or args.model_name == 'fmnist_2nn':
        net = Mnist_2NN()
        if args.model_name == 'fmnist_2nn':
            datasetname = 'fmnist'
        else:
            datasetname = 'mnist'

    elif args.model_name == 'mnist_cnn' or args.model_name == 'fmnist_cnn':
        net = Mnist_CNN()
        if args.model_name == 'fmnist_cnn':
            datasetname = 'fmnist'
        else:
            datasetname = 'mnist'

    elif args.model_name == 'femnist_cnn' or args.model_name == 'femnist500_cnn':
        net = FEMnist_CNN()
        net_counter = FEMnist_CNN()
        net_global = FEMnist_CNN()
        net_local = FEMnist_CNN()
        net_local_personal = FEMnist_CNN()
        datasetname = 'femnist'

    elif args.model_name == 'cifar10_cnn':
        # net = resnet20()
        # net_counter=resnet20()
        # net_global=resnet20()

        # the 24x24 version
        net = Cifar10_CNN()
        datasetname = 'cifar10'
        # net = Cifar10_CNN3()
        # net_counter=Cifar10_CNN3()
        # net_global=Cifar10_CNN3()
        # net_local=Cifar10_CNN3()

        # the 32x32 version
        # net = CifarCnn((3, 32, 32),10)
        # net_counter=CifarCnn((3, 32, 32),10)
        # net_global=CifarCnn((3, 32, 32),10)

    elif args.model_name == 'shakespeare_lstm':
        net = CharLSTM()
        datasetname = 'shakespeare'

    else:
        raise ValueError("Unknown model_name : {}".format(args.model_name))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
        net_counter = torch.nn.DataParallel(net_counter)
        net_global = torch.nn.DataParallel(net_global)

    net = net.to(dev)
    init_params = copy.deepcopy(net.state_dict())   # 初始模型参数
    loss_func = F.cross_entropy  # 交叉熵损失

    """
    现在构造客户端群,每个客户端拥有自己的训练集和测试集,且每次执行保持一致
    需要注意的是各个客户端测试集都是从test_data test_label中挑选的,原测试集(test_data test_label)未被打乱可用来作为Server的测试集
    
    """
    myClients = ClientsGroup(args, datasetname, dev)
    # 对myClients进行保存，方便下次直接加载
    # torch.save(myClients, args.client_path +'{}_{}_myClients.pkl'.format(datasetname, args.beta))
    # myClients = torch.load(args.client_path+'{}_{}_myClients.pkl'.format(datasetname, args.beta))

    opti = optim.SGD(net.parameters(), lr=args.lr)

    # 1. 通过检验各个本地客户端的local accuracy 和 global accuracy来检验前序准备正确

    '''
    # 创建文件保存本地客户端的local accuracy 和 global accuracy
    df = pd.DataFrame(columns=['client', 'accuracy'])
    df.to_csv('实验测试结果/' + 'client_local_accuracy_{}.csv'.format(args.model_name), index=False)
    df = pd.DataFrame(columns=['client', 'accuracy'])
    df.to_csv('实验测试结果/' + 'client_global_accuracy_{}.csv'.format(args.model_name), index=False)
    '''

    # 2. 随机获取（u,S）对

    """
    for loo_client in np.arange(args.num_of_clients):
        all_clients = ['client{}'.format(i) for i in np.arange(args.num_of_clients)]
        # all_clients.remove('client{}'.format(loo_client))
        net.load_state_dict(init_params)
        if args.isolated:
            train_isolated(args, dev, net, myClients, loss_func,opti, all_clients, loo_client)
        else:
            train(args, dev, net, myClients, loss_func,opti, all_clients, loo_client)

    all_clients = ['client{}'.format(i) for i in np.arange(args.num_of_clients)]
    K = 1000
    N = args.num_of_clients
    Pair = {}
    for i in np.arange(K):
        all_clients = ['client{}'.format(i) for i in np.arange(args.num_of_clients)]
        m = np.random.randint(low=1, high=args.num_of_clients-1, dtype="int32")
        u = np.random.randint(N-m, dtype="int32")
        order = list(np.random.permutation(args.num_of_clients))
        S = ['client{}'.format(i) for i in order[0:m]]
        for item in S:
            all_clients.remove(item)
        u = all_clients[u]
        Pair[i] = {u: S}

    Pair = dict_slice(Pair, 0, 1000)

    """

    # 3. 对每一对进行测试,获取sample

    """
    for key in Pair.keys():
        u, S = dict_one(Pair[key])
        print(key, len(S), u)
        all_clients = ['client{}'.format(i) for i in np.arange(args.num_of_clients)]

        # train isolated u
        args.isolated = True
        net.load_state_dict(init_params)
        local_parameters, accuracy_u = train_isolated(args, dev, net, myClients, loss_func, opti, all_clients, loo_client=u)
        print("accuracy_u:", accuracy_u)
        model_u = net.load_state_dict(local_parameters)

        # train_S
        args.isolated = False
        net.load_state_dict(init_params)
        global_parameters_S, accuracy_S = train(args, dev, net, myClients, loss_func, opti, S, loo_client=None)
        print("accuracy_S:", accuracy_S)
        model_S = net.load_state_dict(global_parameters_S)

        # train_Su
        S.append(u)
        net.load_state_dict(init_params)
        global_parameters_Su, accuracy_Su = train(args, dev, net, myClients, loss_func, opti, S, loo_client=None)
        print("accuracy_Su:", accuracy_Su)
        model_Su = net.load_state_dict(global_parameters_Su)

    """

    # 4. 保存sample

    """
        torch.save(
            {
                "S": S,
                "u": u,
                "accuracy_u": accuracy_u,
                "accuracy_S": accuracy_S,
                "accuracy_Su": accuracy_Su,
                "model_u": model_u,
                "model_S": model_S,
                "model_Su": model_Su,
            },
            args.saved_su_model_path+"{}_{}.tar".format(key+1, u)
        )

        checkpoint = torch.load( args.saved_su_model_path+"{}_{}.tar".format(key+1, u))
        print("S:", checkpoint["S"])
        print("u:", checkpoint["u"])
        print("accuracy_u:", checkpoint["accuracy_u"])
        print("accuracy_S:", checkpoint["accuracy_S"])
        print("accuracy_Su:", checkpoint["accuracy_Su"])

    """

    # 5. 对sample进行分类
    """

    with open("分类结果/0.txt", 'a+', encoding='utf-8') as f:
        f.truncate(0)
    with open("分类结果/1.txt", 'a+', encoding='utf-8') as f:
        f.truncate(0)

    file_list = []
    scan_floder(args.saved_su_model_path, file_list)
    for file in file_list:
        checkpoint = torch.load(args.saved_su_model_path+file)
        file = file.rstrip(".tar")
        if checkpoint["accuracy_Su"] > checkpoint["accuracy_S"]:
            with open("分类结果/1.txt", "a") as f:
                f.write(file+"\n")
        else:
            with open("分类结果/0.txt", "a") as f:
                f.write(file+"\n")

    """

    # 6. 训练数据集构造，分别对5个数据集进行构造
    
    file_list = []
    x_train = []
    y_train = []
    scan_floder("Shakespeare_lstm_su_models/", file_list)
    for file in file_list:
        checkpoint = torch.load("fl-svm-1/Shakespeare_lstm_su_models/"+file)
        S = checkpoint["S"]
        S.pop()  # 这里是因为之前计算Su最后加上了u
        u = checkpoint["u"]
        len_S = len(checkpoint["S"])
        num_u = checkpoint["u"].split('t')[-1]
        label = 0
        if checkpoint["accuracy_Su"] > checkpoint["accuracy_S"]:
            label = 1

        # 这里是特征选择，如果将模型参数加入特征之中会导致准确率下降
        '''   
        fc1_w_u = torch.reshape(checkpoint["model_u_para"]['fc1.weight'], (-1, ))
        fc1_b_u = checkpoint["model_u_para"]['fc1.bias']
        fc2_w_u = torch.reshape(checkpoint["model_u_para"]['fc2.weight'], (-1, ))
        fc2_b_u = checkpoint["model_u_para"]['fc2.bias']
        fc3_w_u = torch.reshape(checkpoint["model_u_para"]['fc3.weight'], (-1, ))
        fc3_b_u = checkpoint["model_u_para"]['fc3.bias']
        model_u_para = torch.cat([fc2_b_u, fc3_b_u], dim=0)

        fc1_w_S = torch.reshape(checkpoint["model_S_para"]['fc1.weight'], (-1, ))
        fc1_b_S = checkpoint["model_S_para"]['fc1.bias']
        fc2_w_S = torch.reshape(checkpoint["model_S_para"]['fc2.weight'], (-1, ))
        fc2_b_S = checkpoint["model_S_para"]['fc2.bias']
        fc3_w_S = torch.reshape(checkpoint["model_S_para"]['fc3.weight'], (-1, ))
        fc3_b_S = checkpoint["model_S_para"]['fc3.bias']
        model_S_para = torch.cat([fc2_b_S, fc3_b_S], dim=0)

        ave_accuracy = 0
        for item in S:
            with open("fl-svm-1/log.txt") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if item == line.split(' ')[-1]:
                        num = int(line.split(' ')[0])+1
                        break
            path = "{}_{}.tar".format(num, item)
            it = torch.load("fl-svm-1/su_models/"+path)
            ave_accuracy += it["accuracy_u"]
        ave_accuracy = ave_accuracy/len(S)
        print("ave_accuracy", ave_accuracy)
        '''

        # 最后选择的参数是[int(len_S),  checkpoint["accuracy_u"], checkpoint["accuracy_S"]]
        para_total = torch.cat(
            [torch.tensor([int(len_S),  checkpoint["accuracy_u"], checkpoint["accuracy_S"]])], 0)
        x_train.append(np.asarray(para_total))
        y_train.append(np.asarray(label))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    x_ = x_train
    y_ = y_train

    # 7. 各个分类器分类

    linear_svc_acc = [[1 for j in range(5)] for i in range(20)]
    linear_mean=[]
    linear_var=[]
    linear_min=[]
    linear_max=[]
    poly_svc_acc = [[1 for j in range(5)] for i in range(20)]
    poly_mean=[]
    poly_var=[]
    poly_min=[]
    poly_max=[]
    rbf_svc_acc =  [[1 for j in range(5)] for i in range(20)]
    rbf_mean=[]
    rbf_var=[]
    rbf_min=[]
    rbf_max=[]
    KNN_acc = [[1 for j in range(5)] for i in range(20)]
    KNN_mean=[]
    KNN_var=[]
    KNN_min=[]
    KNN_max=[]
    tree_acc = [[1 for j in range(5)] for i in range(20)]
    tree_mean=[]
    tree_var=[]
    tree_min=[]
    tree_max=[]
    forest_acc = [[1 for j in range(5)] for i in range(20)]
    forest_mean=[]
    forest_var=[]
    forest_min=[]
    forest_max=[]
    LogisticRegression_acc = [[1 for j in range(5)] for i in range(20)]
    LogisticRegression_mean=[]
    LogisticRegression_var=[]
    LogisticRegression_min=[]
    LogisticRegression_max=[]
    Xgboost_acc = [[1 for j in range(5)] for i in range(20)]
    Xgboost_mean=[]
    Xgboost_var=[]
    Xgboost_min=[]
    Xgboost_max=[]
    Transformer_acc = [[1 for j in range(5)] for i in range(20)]
    Transformer_mean=[]
    Transformer_var=[]
    Transformer_min=[]
    Transformer_max=[]

    random=-1
    for random_number in [12,14,16,18,20]:
        random+=1
        x_train, x_test, y_train, y_test = train_test_split( x_, y_, test_size=0.2, random_state=random_number)

        # 标准化
        transfer = StandardScaler()
        x_train = transfer.fit_transform(x_train)
        x_test = transfer.transform(x_test)

        # 降维处理
        # pca1 = PCA(n_components=None, svd_solver='auto')
        # pca1.fit(x_train)
        # x_train = pca1.transform(x_train)
        # pca2 = PCA(n_components=None, svd_solver='auto')
        # pca2.fit(x_test)
        # x_test = pca1.transform(x_test)

        x = x_train
        y = y_train

        # 画出数据点分布

        """
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        for i in range(len(x_train)):
            if y_train[i] == 1:
                x1.append(x_train[i][0])
                y1.append(x_train[i][1])
                z1.append(x_train[i][2])
            else:
                x2.append(x_train[i][0])
                y2.append(x_train[i][1])
                z2.append(x_train[i][2])

        # 设置三维图形模式
        fig = plt.figure()  # 创建一个画布figure,然后在这个画布上加各种元素。
        ax = Axes3D(fig)  # 将画布作用于 Axes3D 对象上。

        ax.scatter(x1, y1, z1, c='r', marker='^')
        ax.scatter(x2, y2, z2, c='g', marker='*')

        ax.set_xlabel('X label')
        ax.set_ylabel('Y label')
        ax.set_zlabel('Z label')

        plt.show()
        """
        
        round=-1
        for i in [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800]:
            round+=1
            x_train = x[:i]
            y_train = y[:i]

            """
            使用三种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据进行预测
            """

            """
            1.使用线性核函数配置的支持向量机进行回归训练并预测
            """
            linear_svc = SVC(kernel='linear',probability=True)
            param_grid = {'C': [0.01,0.03,0.09,0.1, 0.3, 0.9, 1, 3, 9, 10, 30, 90, 100]}
            grid_search = GridSearchCV(linear_svc, param_grid,  refit=True, verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            # 查看具体参数
            # for para, val in list(best_parameters.items()):
            #     print(para, val)
            # print("best_parameters['C']",best_parameters['C'])
            
            linear_svc = SVC(kernel='linear', C=best_parameters['C'],
                        gamma=best_parameters['gamma'], probability=True)
            linear_svc.fit(x_train, y_train.ravel())
            linear_svc_predict = linear_svc.predict(x_test)
            # print("y_test:")
            # print(y_test)
            # print("linear_svc_predict:")
            # print(linear_svc_predict)
            count = 0
            for i in range(len(y_test)):
                if y_test[i] == linear_svc_predict[i]:
                    count += 1
            linear_svc_accuracy = count/len(y_test)
            linear_svc_acc[round][random]=linear_svc_accuracy    
            print('The accuracy of linear SVC', random,round,linear_svc_accuracy)


            """
            2.使用多项式核函数配置的支持向量机进行回归训练并预测
            """
            poly_svc = SVC(kernel='poly',probability=True)
            param_grid = {'C': [0.01,0.03,0.09,0.1, 0.3, 0.9, 1, 3, 9, 10, 30, 90, 100],"degree":[1, 2, 3, 4, 5]}
            grid_search = GridSearchCV(poly_svc, param_grid,  refit=True,verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            # print("best_parameters['C']",best_parameters['C'])
            # print("best_parameters['degree']",best_parameters['degree'])
            
            poly_svc = SVC(kernel='poly', C=best_parameters['C'],degree=best_parameters['degree'], probability=True)
            
            poly_svc.fit(x_train, y_train.ravel())
            poly_svc_predict = poly_svc.predict(x_test)
            count = 0
            for i in range(len(y_test)):
                if y_test[i] == poly_svc_predict[i]:
                    count += 1
            poly_svc_accuracy = count/len(y_test)
            poly_svc_acc[round][random]=poly_svc_accuracy  
            print('The accuracy of poly SVC',random,round, poly_svc_accuracy)


            """
            3.使用高斯核函数配置的支持向量机进行回归训练并预测
            """
            rbf_svc = SVC(kernel='rbf', probability=True)
            param_grid = {'C': [0.1, 0.3, 0.9, 1, 3, 9, 10, 30, 90, 100],'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
            grid_search = GridSearchCV(rbf_svc, param_grid,  refit=True,verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            # print("best_parameters['C']",best_parameters['C'])
            # print("best_parameters['gamma']",best_parameters['gamma'])
            rbf_svc = SVC(kernel='rbf', C=best_parameters['C'],gamma=best_parameters['gamma'], probability=True)
            rbf_svc.fit(x_train, y_train.ravel())
            rbf_svc_predict = rbf_svc.predict(x_test)
            count = 0
            for i in range(len(y_test)):
                if y_test[i] == rbf_svc_predict[i]:
                    count += 1
            rbf_svc_accuracy = count/len(y_test)
            rbf_svc_acc[round][random]=rbf_svc_accuracy  
            print('The accuracy of rbf SVC',random,round, rbf_svc_accuracy)


            """
            4.KNN
            """
            estimator = KNeighborsClassifier()
            param_grid = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9]}
            grid_search = GridSearchCV(estimator, param_grid,  refit=True,verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            # print("best_parameters['n_neighbors']:", best_parameters['n_neighbors'])
            estimator = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'])
            estimator.fit(x_train, y_train.ravel())
            y_predict = estimator.predict(x_test)
            score = estimator.score(x_test, y_test)
            print('The accuracy of KNN', random,round,score)
            KNN_acc[round][random]=score


            """
            5.决策树
            """
            estimator = DecisionTreeClassifier()
            param_grid = {'criterion': ["entropy", "gini"]}
            grid_search = GridSearchCV(estimator, param_grid,  refit=True,verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            # print("best_parameters['criterion']:", best_parameters['criterion'])
            estimator = DecisionTreeClassifier(criterion=best_parameters['criterion'])
            estimator.fit(x_train, y_train.ravel())
            y_predict = estimator.predict(x_test)

            score = estimator.score(x_test, y_test)
            print('The accuracy of decision tree', random,round,score)
            tree_acc[round][random]=score


            """
            6.随机森林
            """
            estimator = RandomForestClassifier()
            param_grid = {'n_estimators': [300, 400, 500, 600],'criterion': ["entropy", "gini"],'max_depth': [4, 6, 8, 10, "None"]}
            grid_search = GridSearchCV(estimator, param_grid,  refit=True,verbose=1, n_jobs=-1, scoring='accuracy', cv=3)
            grid_search.fit(x_train, y_train.ravel())
            best_parameters = grid_search.best_estimator_.get_params()
            estimator = RandomForestClassifier(n_estimators=best_parameters['n_estimators'], criterion=best_parameters['criterion'], max_depth=best_parameters['max_depth'])
            estimator.fit(x_train, y_train.ravel())
            y_predict = estimator.predict(x_test)
            score = estimator.score(x_test, y_test)
            print('The accuracy of random forest', random,round,score)
            forest_acc[round][random]=score

            """
            7.逻辑回归
            """
            estimator = LogisticRegression()
            estimator.fit(x_train, y_train)
            y_predict = estimator.predict(x_test)
            score = estimator.score(x_test, y_test)
            print('The accuracy of LogisticRegression',random,round, score)
            LogisticRegression_acc[round][random]=score

            """
            8 Xgboost
            """
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'error',
                      'lambda': 10,
                      'eta': 0.025,
                      'seed': 0,
                      "reg_alpha": 0,
                      }
            param_grid = {
                'max_depth': [2,3, 4],
                'min_child_weight': [ 2, 3,4],
                'subsample': [0.7, 0.8, 0.9,1],
                'colsample_bytree': [0.6, 0.7, 0.8],
                'gamma': [0.3,0.4,0.5],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [50, 100, 150]
            }

            model = xgb.XGBClassifier(**params)

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

            grid_search.fit(x_train, y_train)

            # Print the best parameters and the corresponding accuracy
            # print("Best Parameters: ", grid_search.best_params_)
            # print("Best Accuracy: ", grid_search.best_score_)

            # xgboost模型初始化设置
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dtest = xgb.DMatrix(x_test)
            watchlist = [(dtrain, 'train')]
            # booster:
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'error',
                      'max_depth': grid_search.best_params_['max_depth'],
                      'lambda': 10,
                      'subsample': grid_search.best_params_['subsample'],
                      'colsample_bytree': grid_search.best_params_['colsample_bytree'],
                      'min_child_weight': grid_search.best_params_['min_child_weight'],
                      'eta': 0.025,
                      'seed': 0,
                      "reg_alpha": 0,
                      'gamma': grid_search.best_params_['gamma'],
                      'learning_rate': grid_search.best_params_['learning_rate'],
                      'n_estimators': grid_search.best_params_['n_estimators']
                      }

            bst = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
            ypred = bst.predict(dtest)

            y_pred = (ypred >= 0.5)*1
            Xgboost_accuracy = metrics.accuracy_score(y_test, y_pred)
            print('The accuracy of Xgboost %.4f' % Xgboost_accuracy)
            Xgboost_acc[round][random]=Xgboost_accuracy

            """
            9 transformer
            """
            # 初始化 Transformer 模型
            model = TransformerModel(input_size=3, num_classes=2)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            num_epochs = 1000
            for epoch in range(num_epochs):
                outputs = model(torch.tensor(x_train))
                loss = criterion(outputs, torch.tensor(y_train).long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                outputs = model(torch.tensor(x_test))
                _, predicted = torch.max(outputs.data, 1)
                transformer_accuracy = (predicted == torch.tensor(y_test)).sum().item() / len(y_test)
                print(f'The accuracy of Transformer {transformer_accuracy:.2f}')
                Transformer_acc[round][random]=transformer_accuracy

    for i in range(20):
        linear_var.append(np.var(linear_svc_acc[i]))
        linear_mean.append(np.mean(linear_svc_acc[i]))
        linear_min.append(np.min(linear_svc_acc[i]))
        linear_max.append(np.max(linear_svc_acc[i]))

        poly_var.append(np.var(poly_svc_acc[i]))
        poly_mean.append(np.mean(poly_svc_acc[i]))
        poly_min.append(np.min(poly_svc_acc[i]))
        poly_max.append(np.max(poly_svc_acc[i]))

        rbf_var.append(np.var(rbf_svc_acc[i]))
        rbf_mean.append(np.mean(rbf_svc_acc[i]))
        rbf_min.append(np.min(rbf_svc_acc[i]))
        rbf_max.append(np.max(rbf_svc_acc[i]))

        KNN_var.append(np.var(KNN_acc[i]))
        KNN_mean.append(np.mean(KNN_acc[i]))
        KNN_min.append(np.min(KNN_acc[i]))
        KNN_max.append(np.max(KNN_acc[i]))

        tree_var.append(np.var(tree_acc[i]))
        tree_mean.append(np.mean(tree_acc[i]))
        tree_min.append(np.min(tree_acc[i]))
        tree_max.append(np.max(tree_acc[i]))

        forest_var.append(np.var(forest_acc[i]))
        forest_mean.append(np.mean(forest_acc[i]))
        forest_min.append(np.min(forest_acc[i]))
        forest_max.append(np.max(forest_acc[i]))

        LogisticRegression_var.append(np.var(LogisticRegression_acc[i]))
        LogisticRegression_mean.append(np.mean(LogisticRegression_acc[i]))
        LogisticRegression_min.append(np.min(LogisticRegression_acc[i]))
        LogisticRegression_max.append(np.max(LogisticRegression_acc[i]))

        Xgboost_var.append(np.var(Xgboost_acc[i]))
        Xgboost_mean.append(np.mean(Xgboost_acc[i]))
        Xgboost_min.append(np.min(Xgboost_acc[i]))
        Xgboost_max.append(np.max(Xgboost_acc[i]))

        Transformer_var.append(np.var(Transformer_acc[i]))
        Transformer_mean.append(np.mean(Transformer_acc[i]))
        Transformer_min.append(np.min(Transformer_acc[i]))
        Transformer_max.append(np.max(Transformer_acc[i]))

    print("linear_svc_acc", linear_svc_acc)
    print("linear_mean", linear_mean)
    print("linear_var", linear_var)
    print("linear_min", linear_min)
    print("linear_max", linear_max)

    print("poly_svc_acc", poly_svc_acc)
    print("poly_mean", poly_mean)
    print("poly_var", poly_var)
    print("poly_min", poly_min)
    print("poly_max", poly_max)

    print("rbf_svc_acc", rbf_svc_acc)
    print("rbf_mean", rbf_mean)
    print("rbf_var", rbf_var)
    print("rbf_min", rbf_min)
    print("rbf_max", rbf_max)

    print("KNN_acc", KNN_acc)
    print("KNN_mean", KNN_mean)
    print("KNN_var", KNN_var)
    print("KNN_min", KNN_min)
    print("KNN_max", KNN_max)

    print("tree_acc", tree_acc)
    print("tree_mean", tree_mean)
    print("tree_var", tree_var)
    print("tree_min", tree_min)
    print("tree_max", tree_max)

    print("forest_acc", forest_acc)
    print("forest_mean", forest_mean)
    print("forest_var", forest_var)
    print("forest_min", forest_min)
    print("forest_max", forest_max)

    print("LogisticRegression_acc", LogisticRegression_acc)
    print("LogisticRegression_mean", LogisticRegression_mean)
    print("LogisticRegression_var", LogisticRegression_var)
    print("LogisticRegression_min", LogisticRegression_min)
    print("LogisticRegression_max", LogisticRegression_max)

    print("Xgboost_acc", Xgboost_acc)
    print("Xgboost_mean", Xgboost_mean)
    print("Xgboost_var", Xgboost_var)
    print("Xgboost_min", Xgboost_min)
    print("Xgboost_max", Xgboost_max)
    
    print("Transformer_acc", Transformer_acc)
    print("Transformer_mean", Transformer_mean)
    print("Transformer_var", Transformer_var)
    print("Transformer_min", Transformer_min)
    print("Transformer_max", Transformer_max)