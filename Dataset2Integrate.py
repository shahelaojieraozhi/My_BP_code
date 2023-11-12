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


class PPG2BPDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, path, mode='train'):
        super(PPG2BPDataset, self).__init__()
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

        """ PyTorch中的tensor可以保存成.pt或者.pth格式的文件, 使用torch.save()方法保存张量, 使用torch.load()来读取张量"""
        # torch.save(self.ppg, "data/ppg.pt")
        # torch.save(self.BP, "data/BP.pt")

        # torch.save(self.ppg, "data/ppg.pth")
        # torch.save(self.BP, "data/BP.pth")

        torch.save(self.ppg, "data/" + mode + "/ppg.h5")
        torch.save(self.BP, "data/" + mode + "/BP.h5")

        # y = torch.load("./myTensor.pt")
        # y = torch.load("./myTensor.pt")

        # self.sbp = (self.BP[:, 0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        # self.dbp = (self.BP[:, 1] - self.DBP_min) / (self.DBP_max - self.DBP_min)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.ppg)


if __name__ == '__main__':
    # data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"
    # data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val"
    data = PPG2BPDataset(data_root_path, mode=data_root_path.split("\\")[-1])
    datasize = len(data)
    print(datasize)
    # pulse, sbp, dbp = data[1]
    # b = data[0]
