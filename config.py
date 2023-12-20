import argparse
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:
    # 'fmnist_2nn'   'cifar10_cnn'   'mnist_2nn'    'femnist_cnn'    'shakespeare_lstm'
    model_name: str = 'femnist_cnn'
    gpu: str = '0'
    method: str = 'fedavg'
    loo_clients = ' '
    memo: str = ' '
    # 'homo'   'iid-diff-quantity'   'noniid-#label4'    'noniid-labeldir'    'noniid-#label2'
    partition: str = "noniid-labeldir"
    beta: float = 0.5
    parallel: bool = True
    isolated: bool = False
    save_full: bool = True

    lr: float = 0.01  # 如果是'shakespeare_lstm'改为0.1
    num_of_clients: int = 100
    num_comm: int = 40
    seed: int = 245

    val_freq: int = 10
    saved_model_name: str = ''
    save_path: str = 'clients_models/'
    client_path: str = 'my_client/'
    saved_su_model_path: str = 'su_models/'

    cfraction: float = 0.1
    epoch: int = 10  # 10
    batchsize: int = 10   # 如果是femnist调为10，其他为128
    name: str = method + '_' + model_name+'_' + partition + '_' + "cf" + str(cfraction)
    client_save_path: str = save_path + name
    start = 0
    end = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loo_clients",
        nargs='+',
        type=int,
        default='0'
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default='0'
    )
    parser.add_argument(
        "--now",
        type=str,
        default='2023_8_5_12_00_00'
    )
    args = parser.parse_args()
    return args
