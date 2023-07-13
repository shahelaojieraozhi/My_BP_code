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

        self.file_count = 0
        path = path
        self.h5_path_list = os.listdir(path)
        with h5py.File(os.path.join(path, self.h5_path_list[self.file_count]), 'r') as f:
            self.ppg = f.get('/ppg')[:].astype(np.float32)
            self.BP = f.get('/label')[:].astype(np.float32)
            self.ppg = torch.from_numpy(self.ppg)
            self.BP = torch.from_numpy(self.BP)

            # subject_idx = f.get('/subject_idx')

    def __getitem__(self, index):
        self.file_count = index // 1000
        index = index - 1000 * self.file_count

        ppg = self.ppg[index, :]
        bp = self.BP[index, :]
        ppg = ppg.unsqueeze(dim=0)
        # ppg = torch.transpose(ppg, 1, 0)

        sbp = (bp[0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        dbp = (bp[1] - self.DBP_min) / (self.DBP_max - self.DBP_min)

        return ppg, sbp, dbp

    def __len__(self):
        return len(self.h5_path_list) * 1000


if __name__ == '__main__':
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    data = PPG2BPDataset(data_root_path)
    # datasize = len(data)
    pulse, bp_label = data[1001]
    b = data[0]

"""
SBP_min = 40
SBP_max = 200
DBP_min = 40
DBP_max = 120
"""
