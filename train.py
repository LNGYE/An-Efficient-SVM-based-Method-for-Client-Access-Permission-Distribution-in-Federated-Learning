import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from tqdm import tqdm
from utils import mkdir_if_needed, norm_grad

from clients import ClientsGroup


def train(args, dev, net, myClients, loss_func, opti, all_clients, loo_client=None):
    # net是初始模型
    # 这个testdataloader 里面的测试数据是没有打乱的总的测试数据
    testDataLoader = myClients.test_data_loader
    # 选择0.1比例客户端来通讯
    num_in_comm = int(max(len(all_clients) * args.cfraction, 1))
    global_parameters = {}

    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args.num_comm):
        net.train()
        print("communicate round {}".format(i+1))
        order = list(np.random.permutation(len(all_clients)))

        # 需要注意的一点是loo_client可以是None,当不是None的时候说明此时用的是n-1个客户端进行全局训练，当是None的时候则是用n个客户端进行全局训练
        if loo_client is not None:
            order.remove(loo_client)
        clients_in_comm = [all_clients[i] for i in order[0:num_in_comm]]
        sum_parameters = None
        for ith, client in enumerate(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(net, client, loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 更新全局模型参数
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        # 达到验证频率,计算更新后的全局模型的准确率
        # if (i + 1) % args.val_freq == 0:
        if (i + 1) == args.num_comm:
            with torch.no_grad():
                net.load_state_dict(global_parameters, strict=True)
                net.eval()
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1

                # 下面这段代码是用于计算每个客户端在测试集上的准确率，并根据准确率排序找出准确率最高的 10% 和准确率最低的 10% 的客户端，并计算它们的平均准确率
            
                """
                accu_by_client = []
                for client in tqdm(all_clients):
                    num_client = 0
                    sum_accu_client = 0

                    test_dl = myClients.clients_set[client].test_dl
                    net.eval()
                    for data, label in test_dl:
                        data, label = data.to(dev), label.to(dev)
                        preds = net(data)
                        preds = torch.argmax(preds, dim=1)
                        sum_accu_client += (preds == label).float().mean()
                        num_client += 1

                    accu_by_client.append(sum_accu_client / num_client)

                accu_by_client = torch.Tensor(accu_by_client)
                sorted, indices = torch.sort(accu_by_client, descending=True)
                acc_best_10 = sorted[0:int(0.1*args.num_of_clients)].mean()
                acc_worst_10 = sorted[-int(0.1*args.num_of_clients):].mean()
                wandb.log({"acc": sum_accu / num, 
                            'std': accu_by_client.std(dim=0), 
                            'worst': min(accu_by_client), 
                            'worst10%': acc_worst_10, 
                            'best': max(accu_by_client), 
                            'best10%': acc_best_10})
                print('accuracy: {}'.format(sum_accu / num))

                """

    """
    if loo_client is not None:
        # 这里保存的模型是除去loo_client进行全局训练得到的全局模型
        net.load_state_dict(global_parameters, strict=True)
        torch.save(net, args.client_save_path + '/' + "loo_client{}.pth".format(loo_client))

    else:
        # 这里保存的是每个客户端训练后的本地模型
        for client in tqdm(all_clients):
            net.load_state_dict(myClients.clients_set[client].local_parameters, strict=True)
            torch.save(net, args.client_save_path +'/' + "{}.pth".format(client))
        # 这里保存的是用所有客户端训练后得到的全局模型
        net.load_state_dict(global_parameters, strict=True)
        torch.save(net, args.client_save_path + '/' + "full.pth")
    """

    return global_parameters, (sum_accu / num).item()


def train_isolated(args, dev, net, myClients, loss_func, opti, all_clients, loo_client=None):
    # loo_client是被独立出来的客户端
    # 这个testdataloader 里面的测试数据是没有打乱的总的测试数据
    net.train()
    testDataLoader = myClients.test_data_loader
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    client = loo_client
    local_parameters = myClients.clients_set[client].localUpdate(net, client, loss_func, opti, global_parameters, isolated=args.isolated)

    with torch.no_grad():
        net.load_state_dict(local_parameters, strict=True)
        net.eval()
        sum_accu = 0
        num = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1

        global_accuracy = sum_accu / num
        client = client
        list = [client, global_accuracy.item()]
        data = pd.DataFrame([list])
        # data.to_csv('fl-svm-1/实验测试结果/' + 'client_global_accuracy_{}.csv'.format(args.model_name),mode='a', header=False, index=False)

    net.load_state_dict(local_parameters, strict=True)
    # 保存了独立客户端模型
    # torch.save(net, args.client_save_path + '/' + "isolated_client{}.pth".format(loo_client))
    return local_parameters, global_accuracy.item()