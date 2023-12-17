# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/16 17:37
@Author  : Rao Zhi
@File    : inference.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import matplotlib.pyplot as plt
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


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To prohibit hash randomization and make the experiment replicable
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

ppgs = torch.load("ppg.h5")
labels = torch.load("label.h5")

model = resnet18(input_c=1, num_classes=1)
model = model.to(device)

model_name = 'abnormal_pulse_detection_resnet-18_2023121618'
model.load_state_dict(torch.load('save/' + model_name + '/best_w.pth')['state_dict'])
best_epoch = torch.load('save/' + model_name + '/best_w.pth')["epoch"]
print(f"best epoch:{best_epoch}")

model.eval()
# ppg = ppg[1200]
# label = labels[1200]
pre = []
with torch.no_grad():
    for i in range(100):
        # ppg = ppgs[i + 1200].unsqueeze(dim=0).unsqueeze(dim=0)
        b = ppgs[1400 + i]
        ppg = ppgs[1400 + i].unsqueeze(dim=0).unsqueeze(dim=0)
        ppg = ppg.to(device)

        a = model(ppg).cpu().squeeze()
        output = torch.sigmoid(model(ppg).cpu().squeeze())
        predict = (output >= 0.5).float()  # 二分类阈值为0.5
        pre.append(predict)
        plt.plot(ppg.squeeze().cpu())
        plt.title(f"predict:{predict}, gt:{labels[1200 + i]}")
        # plt.title(f"predict:{predict}, gt:{labels[i]}")
        plt.show()

# print(pre.count(1))   # show count  of one in list


"""
大问题啊，为啥单个做 inference 跟用predict不一样？
"""
