# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/10 14:50
@Author  : Rao Zhi
@File    : PPG2BP_Dataset_repair_vision.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


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

    def __init__(self, path):
        super(PPG2BPDataset, self).__init__()
        self.SBP_min = 40
        self.SBP_max = 200
        self.DBP_min = 40
        self.DBP_max = 120
        self.h5_path_list = os.listdir(path)

        self.ppg = []
        self.BP = []
        for h5_path in self.h5_path_list:
            with h5py.File(os.path.join(path, h5_path), 'r') as f:
                ppg = f.get('/ppg')[:].astype(np.float32)
                bp = f.get('/label')[:].astype(np.float32)
                self.ppg.append(ppg)
                self.BP.append(bp)
        self.ppg = np.concatenate(self.ppg, axis=0)
        self.BP = np.concatenate(self.BP, axis=0)
        self.ppg = torch.from_numpy(self.ppg)
        self.BP = torch.from_numpy(self.BP)

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
    # data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"
    data = PPG2BPDataset(data_root_path)
    datasize = len(data)
    # pulse, sbp, dbp = data[1]
    # b = data[0]
