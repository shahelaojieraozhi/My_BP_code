# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/17 16:35
@Author  : Rao Zhi
@File    : data_split.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import torch
from sklearn.model_selection import train_test_split

train_ppgs = torch.load("../data_normal/train/ppg.h5")
val_ppgs = torch.load("../data_normal/val/ppg.h5")
test_ppgs = torch.load("../data_normal/test/ppg.h5")

train_labels = torch.load("../data_normal/train/BP.h5")
val_labels = torch.load("../data_normal/val/BP.h5")
test_labels = torch.load("../data_normal/test/BP.h5")

X = torch.cat((train_ppgs, val_ppgs, test_ppgs))
y = torch.cat((train_labels, val_labels, test_labels))

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 / 3, random_state=666)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=666)

# 打印各个集合的大小
print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))

mode = ['train', 'val', 'test']
for element in mode:
    dir_file_path = "dataset/" + element
    os.makedirs(dir_file_path, exist_ok=True)

torch.save(X_train, "dataset/train/" + "ppg.h5")
torch.save(X_val, "dataset/val/" + "ppg.h5")
torch.save(X_test, "dataset/test/" + "ppg.h5")

torch.save(y_train, "dataset/train/" + "BP.h5")
torch.save(y_val, "dataset/val/" + "BP.h5")
torch.save(y_test, "dataset/test/" + "BP.h5")
