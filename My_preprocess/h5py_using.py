# -*- coding: utf-8 -*-
"""
@Project ：Blood_P 
@Time    : 2023/7/7 11:22
@Author  : Rao Zhi
@File    : h5py_using.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 创建HDF5文件
# file = h5py.File('data_new.h5', 'w')
#
# # 创建数据集
# ppg_signal = np.random.random(875)  # 假设数据的shape为(875, 2, 1)
# bp = np.random.random(2)  # 假设数据的shape为(875, 2, 1)
# subject_id = np.array([11])
# dataset1 = file.create_dataset('ppg_signal', data=ppg_signal)
# dataset2 = file.create_dataset('bp', data=bp)
# dataset3 = file.create_dataset('subject_idx', data=subject_id)
#
# # # 保存数据集
# file.close()

# data = {'ppg': ppg_signal, 'label': bp, 'subject_idx': subject_id}
# dataset = file.create_dataset('subject', data=data)
#
# # 保存数据集
# file.close()

# with h5py.File('data_new.h5', 'r') as f:
#     # load ppg and BP data as well as the subject numbers the samples belong to
#
#     ppg_h5 = f.get('/ppg_signal')
#     BP = f.get('/bp')
#     subject_idx = f.get('/subject_idx')
#
#     # 现在还是  HDF5 dataset  形式
#     print(BP)
#     print(subject_idx)
#
#     # 取数了
#     print(ppg_h5[:])
#     print(BP[:])
#     print(subject_idx[:])


with h5py.File('h5_record/test/MIMIC_III_ppg_test_00001_of_00250.h5', 'r') as f:
    # load ppg and BP data as well as the subject numbers the samples belong to

    ppg_h5 = f.get('/ppg')
    BP = f.get('/label')
    subject_idx = f.get('/subject_idx')

    # 现在还是  HDF5 dataset  形式
    # print(BP)
    # print(subject_idx)

    # 取数了
    print(ppg_h5[1, :])

    plt.plot(ppg_h5[1, :])
    plt.show()
    # print(BP.shape)
    # print(subject_idx.shape)

