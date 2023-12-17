# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/10 14:50
@Author  : Rao Zhi
@File    : PPG2BP_Dataset_repair_vision.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def use_derivative(x_input, fs=125):
    """
    X_input: (None, 1, 875)
    fs     : 125
    """
    dt1 = (x_input[:, :, 1:] - x_input[:, :, :-1]) * fs  # (None, 874, 1)  ———pad———> (None, 875, 1)
    dt2 = (dt1[:, :, 1:] - dt1[:, :, :-1]) * fs  # (None, 873, 1)  ———pad———> (None, 875, 1)

    # under padding
    padded_dt1 = torch.nn.functional.pad(dt1, (0, 1, 0, 0), value=0)
    padded_dt2 = torch.nn.functional.pad(dt2, (0, 2, 0, 0), value=0)
    # (0, 0, 0, 1) Indicates the fill size of the left, right, top and bottom edges, respectively

    x = torch.cat([x_input, padded_dt1, padded_dt2], dim=1)
    return x


class PPG2BPDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, mode):
        super(PPG2BPDataset, self).__init__()
        self.SBP_min = 40
        self.SBP_max = 200
        self.DBP_min = 40
        self.DBP_max = 120

        # self.ppg = torch.load("data/" + mode + "/ppg.h5")
        # self.BP = torch.load("data/" + mode + "/BP.h5")

        self.ppg = torch.load("data_normal/" + mode + "/ppg.h5")
        self.BP = torch.load("data_normal/" + mode + "/BP.h5")

        # self.sbp = (self.BP[:, 0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        # self.dbp = (self.BP[:, 1] - self.DBP_min) / (self.DBP_max - self.DBP_min)
        self.sbp = self.BP[:, 0]
        self.dbp = self.BP[:, 1]

    def __getitem__(self, index):
        ppg = self.ppg[index, :]
        ppg = ppg.unsqueeze(dim=0)
        # ppg = torch.transpose(ppg, 1, 0)
        # bp = self.BP[index, :]

        sbp = self.sbp[index]
        dbp = self.dbp[index]

        # sbp = (bp[0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        # dbp = (bp[1] - self.DBP_min) / (self.DBP_max - self.DBP_min)

        return ppg, sbp, dbp

    def __len__(self):
        # return len(self.h5_path_list) * 1000
        return len(self.ppg)


if __name__ == '__main__':
    data = PPG2BPDataset(mode='test')
    datasize = len(data)
    print(datasize)
    for sample in data:
        pulse, sbp, dbp = sample
        plt.plot(pulse.squeeze())
        plt.show()
    # b = data[0]
