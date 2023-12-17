# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/15 16:33
@Author  : Rao Zhi
@File    : generate_data.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train_ppg = torch.load("../data_normal/train/ppg.h5")
test_ppg = torch.load("../data_normal/test/ppg.h5")

train_ = pd.read_csv("train_pulse_detection.csv")
test_ = pd.read_csv("test_pulse_detection.csv")
# train_ = pd.read_csv("modified_train_pulse_detection.csv")
# test_ = pd.read_csv("modified_test_pulse_detection.csv")
train_idx, train_label = train_["idx"], train_["label"]
test_idx, test_label = test_["idx"], test_["label"]

selected_train_pulse = train_ppg[train_idx, :]
selected_test_pulse = test_ppg[test_idx, :]
all_ppg_pulse = torch.cat((selected_train_pulse, selected_test_pulse))
print()
all_label = torch.cat((torch.tensor(np.array(train_label)), torch.tensor(np.array(test_label))))
# a = all_label.data[:300]

# show
# for pulse, label in zip(all_ppg_pulse, all_label):
#     plt.plot(pulse)
#     plt.title(f"{label}")
#     plt.show()

# for i in range(10):
#     pulse = all_ppg_pulse[210 + i]
#     label = all_label[210 + i]
#     plt.plot(pulse)
#     plt.title(f"{label}")
#     plt.show()

torch.save(all_ppg_pulse, "ppg.h5")
torch.save(all_label, "label.h5")
