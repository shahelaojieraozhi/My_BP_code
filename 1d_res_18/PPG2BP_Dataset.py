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
        # if index % 1000 == 0 and index != 0:
        #     self.file_count += 1

        self.file_count = index // 1000
        index = index - 1000 * self.file_count

        pulse = self.ppg[index, :]
        BP = self.BP[index, :]

        return pulse, BP

    def __len__(self):
        return len(self.h5_path_list) * 1000


if __name__ == '__main__':
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    data = PPG2BPDataset(data_root_path)
    # datasize = len(data)
    pulse, bp_label = data[1001]
    b = data[0]
