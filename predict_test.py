# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/11 14:45
@Author  : Rao Zhi
@File    : predict_test.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import shutil
import time
import argparse

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from model.resnet18_1D import resnet18_1d
from PPG2BP_Dataset_v2 import PPG2BPDataset
from model.Resnet import resnet50, resnet34, resnet18, resnet101, resnet152

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)


def inv_normalize(sbp_arr, dbp_arr):
    sbp_min = 40
    sbp_max = 200
    dbp_min = 40
    dbp_max = 120

    sbp_arr = sbp_arr * (sbp_max - sbp_min) + sbp_min
    dbp_arr = dbp_arr * (dbp_max - dbp_min) + dbp_min

    return sbp_arr, dbp_arr


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=2048, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='cnn', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    opt = parser.parse_args()

    print('loading data...')
    test_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"
    test_data = PPG2BPDataset(test_data_path)
    test_loader = DataLoader(test_data, batch_size=opt.batch, shuffle=True, num_workers=0)

    "model"
    resnet_1d = resnet18_1d()
    # resnet_1d = resnet50()
    model = resnet_1d.to(device)
    # model.load_state_dict(torch.load('save/cnn_202307111750/best_w.pth')['state_dict'])     # 18
    # model.load_state_dict(torch.load('save/cnn_202307120933/best_w.pth')['state_dict'])     # 34
    model.load_state_dict(torch.load('save/resnet18_val_loss_2.807/best_w.pth')['state_dict'])  # 50

    model.eval()
    loss_meter, it_count = 0, 0
    test_batch_idx = 0

    with torch.no_grad():
        for (ppg, sbp, dbp) in test_loader:
            ppg = ppg.to(device)
            bp_hat = model(ppg).cpu()
            sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]

            dbp_hat_arr = dbp_hat.numpy()
            sbp_hat_arr = sbp_hat.numpy()

            sbp_arr = sbp.numpy()
            dbp_arr = dbp.numpy()

            sbp_arr, dbp_arr = inv_normalize(sbp_arr, dbp_arr)
            sbp_hat_arr, dbp_hat_arr = inv_normalize(sbp_hat_arr, dbp_hat_arr)

            table_arr = np.vstack((sbp_hat_arr, dbp_hat_arr, sbp_arr, dbp_arr)).T
            pd.DataFrame(table_arr).to_csv(
                "./predict_test/resnet18_val_loss_2.807/predict_test_{}.csv".format(test_batch_idx),
                header=['sbp_hat_arr', 'dbp_hat_arr', 'sbp_arr', 'dbp_arr'], index=False)

            loss_sbp = F.mse_loss(sbp_hat, sbp)
            loss_dbp = F.mse_loss(dbp_hat, dbp)

            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            it_count += 1
            test_batch_idx += 1

    return loss_meter / it_count


if __name__ == '__main__':
    test_loss = test()
    print(test_loss)

# tensorboard --logdir=resnet18_202307141720 --port=6007
# tensorboard --logdir=add_normal_res_18 --port=6007
