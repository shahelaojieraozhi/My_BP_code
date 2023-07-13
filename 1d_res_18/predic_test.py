# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/11 14:45
@Author  : Rao Zhi
@File    : predic_test.py
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
from resnet18_1D import resnet18_1d
from PPG2BP_Dataset import PPG2BPDataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=2048, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='cnn', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    opt = parser.parse_args()

    print('loading data...')
    test_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"
    test_data = PPG2BPDataset(test_data_path)
    test_loader = DataLoader(test_data, batch_size=opt.batch, shuffle=True, num_workers=1)

    "model"
    resnet_1d = resnet18_1d()
    model = resnet_1d.to(device)

    model.eval()
    loss_meter, it_count = 0, 0
    with torch.no_grad():
        for (ppg, bp) in test_loader:
            ppg = ppg.to(device)
            ppg = ppg.unsqueeze(dim=0)
            ppg = torch.transpose(ppg, 1, 0)
            bp_hat = model(ppg).cpu()
            dbp_hat, sbp_hat = bp_hat[:, 0], bp_hat[:, 1]

            dbp_hat_arr = dbp_hat.numpy()
            sbp_hat_arr = sbp_hat.numpy()

            sbp_arr = bp[:, 0].numpy()
            dbp_arr = bp[:, 1].numpy()

            table_arr = np.vstack(dbp_hat_arr, sbp_hat_arr, sbp_arr, dbp_arr)
            pd.DataFrame(table_arr).to_csv("predict_test.csv",
                                           header=['dbp_hat_arr', 'sbp_hat_arr', 'sbp_arr', 'dbp_arr'], index=False)

            loss_dbp = F.mse_loss(dbp_hat, bp[:, 0])
            loss_sbp = F.mse_loss(sbp_hat, bp[:, 1])

            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            it_count += 1

    return loss_meter / it_count


if __name__ == '__main__':
    test()

# tensorboard --logdir=cnn_202305061217 --port=6007
