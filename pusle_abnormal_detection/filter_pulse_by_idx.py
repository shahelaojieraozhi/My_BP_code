# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/17 11:51
@Author  : Rao Zhi
@File    : filter_pulse_by_idx.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch
import pandas as pd

mode = 'train'
# mode = 'val'

# out_domain infer
labels = torch.load("../data_normal/" + mode + "/BP.h5")
ppgs = torch.load("../data_normal/" + mode + "/ppg.h5")

# filter_idx_df = pd.read_csv("train_filter_idx.csv")
filter_idx_df = pd.read_csv(mode + "_filter_idx.csv")
zero_indices = filter_idx_df[filter_idx_df.values.reshape(-1) == 0].index
selected_pulses = ppgs[zero_indices]
selected_labels = labels[zero_indices]

torch.save(selected_pulses, mode + "_selected_ppg.h5")
torch.save(selected_labels, mode + "_selected_label.h5")
