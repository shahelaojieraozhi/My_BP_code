# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/11/13 15:00
@Author  : Rao Zhi
@File    : Data_refresh.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


def filters_normal_bp(bp, ppg):
    # index of row
    sbp_index = 0
    dbp_index = 1

    # define the bound of normal bp value
    lower_sbp_bound = 75
    upper_sbp_bound = 165

    lower_dbp_bound = 40
    upper_dbp_bound = 80

    # Use condition index(条件索引)
    filter_sbp = bp[
        (bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]      # 952   # Separate filtering
    filter_dbp = filter_sbp[
        (filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]  # 881

    filter_sbp_ppg = ppg[
        (bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]      # 952  Separate filtering
    filter_dbp_ppg = filter_sbp_ppg[
        (filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]  # 881
    return filter_dbp, filter_dbp_ppg


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

        self.BP, self.ppg = filters_normal_bp(self.BP, self.ppg)

        self.ppg = torch.from_numpy(self.ppg)
        self.BP = torch.from_numpy(self.BP)
        """ PyTorch中的tensor可以保存成.pt或者.pth格式的文件, 使用torch.save()方法保存张量, 使用torch.load()来读取张量"""
        # torch.save(self.ppg, "data/ppg.pt")
        # torch.save(self.BP, "data/BP.pt")

        # torch.save(self.ppg, "data/ppg.pth")
        # torch.save(self.BP, "data/BP.pth")

        os.makedirs("data_normal/" + mode, exist_ok=True)
        torch.save(self.ppg, "data_normal/" + mode + "/ppg.h5")
        torch.save(self.BP, "data_normal/" + mode + "/BP.h5")

        # y = torch.load("./myTensor.pt")
        # y = torch.load("./myTensor.pt")

        # self.sbp = (self.BP[:, 0] - self.SBP_min) / (self.SBP_max - self.SBP_min)
        # self.dbp = (self.BP[:, 1] - self.DBP_min) / (self.DBP_max - self.DBP_min)

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.ppg)


if __name__ == '__main__':
    data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"       # 218617
    # data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"    # 875554
    # data_root_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val"      # 218022
    data = PPG2BPDataset(data_root_path, mode=data_root_path.split("\\")[-1])
    datasize = len(data)
    print(datasize)
    # pulse, sbp, dbp = data[1]
    # b = data[0]
