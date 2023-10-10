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

import torch

inputs = torch.rand(1024, 874, 1)
padded_x = torch.nn.functional.pad(inputs, (0, 0, 0, 2), value=0)

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
print(padded_x.shape)
