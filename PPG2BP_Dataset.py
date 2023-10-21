# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/10 14:50
@Author  : Rao Zhi
@File    : PPG2BP_Dataset.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def normalization(ori_data):
    return scaler.fit_transform(ori_data)


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

    def __init__(self, path, fs=125, using_derivative=True):
        super(PPG2BPDataset, self).__init__()

        # self.SBP_min = 40
        # self.SBP_max = 200
        # self.DBP_min = 40
        # self.DBP_max = 120
        self.using_derivative = using_derivative
        self.fs = fs
        self.file_count = 0
        path = path
        self.h5_path_list = os.listdir(path)
        with h5py.File(os.path.join(path, self.h5_path_list[self.file_count]), 'r') as f:
            self.ppg = f.get('/ppg')[:].astype(np.float32)
            # self.ppg = normalization(self.ppg)
            self.BP = f.get('/label')[:].astype(np.float32)
            self.ppg = torch.from_numpy(self.ppg)
            self.BP = torch.from_numpy(self.BP)

            # subject_idx = f.get('/subject_idx')

    def __getitem__(self, index):
        # if index % 1000 == 0 and index != 0:
        #     self.file_count += 1

        self.file_count = index // 1000
        index = index - 1000 * self.file_count

        ppg_pulse = self.ppg[index, :]
        bp = self.BP[index, :]

        """ if normalization ?"""
        # sbp = (bp[0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        # dbp = (bp[1] - self.DBP_min) / (self.DBP_max - self.DBP_min)

        return ppg_pulse, bp

    def __len__(self):
        return len(self.h5_path_list) * 1000


if __name__ == '__main__':
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    data = PPG2BPDataset(data_root_path)
    # datasize = len(data)
    pulse, bp_label = data[1001]
    b = data[0]
