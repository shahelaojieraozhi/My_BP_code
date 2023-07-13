# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/10 12:53
@Author  : Rao Zhi
@File    : read_h5.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train\\MIMIC_III_ppg_train_00001_of_01000.h5"
with h5py.File(h5_path, 'r') as f:
    # load ppg and BP data as well as the subject numbers the samples belong to

    ppg_h5 = f.get('/ppg')
    BP = f.get('/label')
    subject_idx = f.get('/subject_idx')

    # 现在还是  HDF5 dataset  形式
    print(ppg_h5)        # shape (1000, 875)
    print(BP)            # shape (1000, 2)
    print(subject_idx)   # shape (1000,)

    # plt.plot(ppg_h5[0, :])
    # plt.show()

    np.set_printoptions(suppress=True)  # 不用科学计数法显示数字

    print(np.array(subject_idx[:]))
