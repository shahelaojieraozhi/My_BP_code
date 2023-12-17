# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/16 21:57
@Author  : Rao Zhi
@File    : predict.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
from torch.utils.data import Dataset
import datetime
import os
import random
import shutil
import time
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
from torch import optim
import utils
import sys
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model.Resnet import resnet18, resnet34, resnet50, resnet101, resnet152

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, mode="train"):
        super(PPGDataset, self).__init__()

        self.ppg = torch.load("ppg.h5")
        self.label = torch.load("label.h5")

        if mode == 'train':
            self.ppg = self.ppg[:1200]
            self.label = self.label[:1200]

        else:
            self.ppg = self.ppg[1200:]
            self.label = self.label[1200:]

    def __getitem__(self, index):
        ppg = self.ppg[index, :]
        ppg = ppg.unsqueeze(dim=0)
        target = self.label[index]

        return ppg, target

    def __len__(self):
        # return len(self.h5_path_list) * 1000
        return len(self.ppg)


def predict(opt):
    "load param"

    "load model"
    # model = resnet50(input_c=1 if input_channel == 1 else 3, num_classes=2)
    model = resnet18(input_c=1, num_classes=1)

    model_name = 'abnormal_pulse_detection_resnet-18_2023121618'
    model.load_state_dict(torch.load('save/' + model_name + '/best_w.pth')['state_dict'])
    best_epoch = torch.load('save/' + model_name + '/best_w.pth')["epoch"]
    print(f"best epoch:{best_epoch}")
    model = model.to(device)

    """load data"""
    print('loading data...')
    train_data = PPGDataset('train')
    val_data = PPGDataset('val')
    print(f"The size of data is:{len(val_data)}")

    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=False, num_workers=0)

    model.eval()
    prediction_res = []
    acc_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for (ppg, label) in val_loader:
            ppg = ppg.to(device)
            a = model(ppg).cpu().squeeze()
            output = torch.sigmoid(model(ppg).cpu().squeeze())

            loss = criterion(output, label.float())

            prediction = (output >= 0.5).float()

            loss_meter += loss.item()
            # predict = output.argmax(dim=1)
            acc = torch.eq(prediction, label).sum().float().item() / len(ppg)
            acc_meter += acc
            it_count += 1
            prediction_res.append(prediction)
        print(f"accuracy of val dataset : {acc_meter / it_count}")

    print(prediction_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model", type=str, default='abnormal_pulse_detection', help="model type")
    parser.add_argument("-d", "--describe", type=str, default='resnet-18', help="describe for this model")
    parser.add_argument("-n", "--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=128, help="batch size of training")
    parser.add_argument("-bl", "--best_loss", type=int, default=1e3, help="best_loss")
    parser.add_argument("-lr", "--lr", type=int, default=1e-4, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=0, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument("-ds", "--decay_step", type=list, default=[100], help="decay step list of learning rate")
    parser.add_argument("-wd", "--weight_decay", type=int, default=1e-3, help="weight_decay")
    parser.add_argument('--using_derivative', default=False, help='using derivative of PPG or not')
    parser.add_argument('--show_interval', type=int, default=3, help='how long to show the loss value')
    parser.add_argument('--loss_func', type=str, default='HuberLoss',
                        choices=('SmoothL1Loss', 'mse', 'bp_bucketing_loss', 'HuberLoss'),
                        help='which loss function is selected')
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="para of HuberLoss and SmoothL1Loss")
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    predict(args)
