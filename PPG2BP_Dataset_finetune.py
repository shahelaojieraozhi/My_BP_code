# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/18 17:23
@Author  : Rao Zhi
@File    : PPG2BP_Dataset_finetune.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import numpy as np
import torch
from torch.utils.data import Dataset


class TDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, x, y):
        super(TDataset, self).__init__()
        self.x = torch.from_numpy(x.astype(np.float32)).unsqueeze(dim=0)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.ppg = torch.transpose(self.x, 1, 0)
        self.sbp = self.y[:, 0]
        self.dbp = self.y[:, 1]

    def __getitem__(self, index):
        return self.ppg[index], self.sbp[index], self.dbp[index]

    def __len__(self):
        return len(self.ppg)
