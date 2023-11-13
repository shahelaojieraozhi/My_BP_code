# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/22 9:42
@Author  : Rao Zhi
@File    : Temp_test.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

# import torch
#
# A = torch.arange(24).reshape(2, 3, 4)
# print(A.shape)
#
# B = A.transpose(2, 1)
# print(B.shape)

# import numpy as np
# data = np.load('08-11-2023_pers_dataset.npz')
# # a = data[0]
# print(data['arr_0'].shape)  # (250000, 875)
# print(data['arr_1'].shape)  # (250000, 2)
# print(data['arr_2'].shape)  # (250000,)

# import torch
#
# print(torch.cuda.current_device())  # 返回当前设备索引
# print(torch.cuda.device_count())  # 返回GPU的数量
# print(torch.cuda.get_device_name(0))  # 返回gpu名字，设备索引默认从0开始
# print(torch.cuda.is_available())  # cuda是否可用
# print(torch.version.cuda)  # cuda版本

# import torch
#
# inputs = torch.rand(1024, 874, 1)
# padded_x = torch.nn.functional.pad(inputs, (0, 0, 0, 2), value=0)


# # 创建一个示例张量
# x = torch.tensor([[1, 2], [3, 4]])
#
# # 使用 pad 函数添加填充
# # 第一个参数是输入张量，第二个参数是填充的大小（一个元组，各个维度的填充数），第三个参数是填充的值
# padded_x = torch.nn.functional.pad(x, (1, 2, 1, 2), value=0)
# """
# 填充的大小参数是一个元组，包含了四个值，分别表示左边、右边、上边和下边的填充大小。
# 在示例中，(1, 2, 1, 2) 表示在左边添加 1 列，右边添加 2 列，上边添加 1 行，下边添加 2 行的填充。
# """
# print(padded_x.shape)


# print("aad", end='\n')
# print("aadxx")
import datetime

import numpy as np
import torch

# x = torch.tensor([1, 2, 3])
# xx = x.repeat(3, 2, 1)  # 将一维度的x向量扩展到三维
# print(xx)

# import torch
# x = torch.tensor([1, 2, 3])
# x1 = x.repeat(3)
# print("x1:\n", x1)
# x2 = x.repeat(3, 1)
# print("x2:\n", x2)
# x3 = x.repeat(3, 2)
# print("x3:\n", x3)
# x4 = x.repeat(3, 2, 1)
# print("x4:\n", x4)


# import torch
#
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# x1 = x.repeat_interleave(3, 0)
# print("x1:\n", x1)
#
# x2 = x.repeat_interleave(3, 1)
# print("x2:\n", x2)


# SBP_min = 4
# SBP_max = 12
# DBP_min = 4
# DBP_max = 12
#
# bp = np.arange(12).reshape(6, 2)
# sbp = (bp[:, 0] - SBP_min) / (SBP_max - SBP_min)
# dbp = (bp[:, 1] - DBP_min) / (DBP_max - DBP_min)

# input_channel = 3
# a = 1 if input_channel == 1 else 3
# print(a)

# print(datetime.datetime.now())

import h5py

path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val\\MIMIC_III_ppg_val_00001_of_00250.h5"

# # 列索引，这里假设你想要处理第二列（索引为1）
sbp_index = 0
dbp_index = 1

# # 定义正常值的范围
lower_sbp_bound = 75
upper_sbp_bound = 165

lower_dbp_bound = 40
upper_dbp_bound = 80

with h5py.File(path, 'r') as f:
    ppg = np.array(f.get('/ppg')[:].astype(np.float32))
    bp = np.array(f.get('/label')[:].astype(np.float32))

# Use condition index(条件索引)
filter_sbp = bp[(bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]                  # 952   # Separate filtering
filter_dbp = filter_sbp[(filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]    # 881

filter_sbp_ppg = ppg[(bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]             # 952  Separate filtering
filter_dbp_ppg = filter_sbp_ppg[(filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]  # 881
print()

# # 创建一个示例数组
# data = np.array([[1, 80, 3],
#                  [2, 90, 5],
#                  [3, 120, 2],
#                  [4, 200, 8],
#                  [5, 150, 6]])
#
# # 列索引，这里假设你想要处理第二列（索引为1）
# column_index = 1
#
# # 定义正常值的范围
# lower_bound = 75
# upper_bound = 165
#
# # 使用条件索引来剔除异常值
# filtered_data = data[(data[:, column_index] >= lower_bound) & (data[:, column_index] <= upper_bound)]
#
# print("原始数据:\n", data)
# print("\n剔除异常值后的数据:\n", filtered_data)
