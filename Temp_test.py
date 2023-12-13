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

# import h5py
#
# path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val\\MIMIC_III_ppg_val_00001_of_00250.h5"
#
# # # 列索引，这里假设你想要处理第二列（索引为1）
# sbp_index = 0
# dbp_index = 1
#
# # # 定义正常值的范围
# lower_sbp_bound = 75
# upper_sbp_bound = 165
#
# lower_dbp_bound = 40
# upper_dbp_bound = 80
#
# with h5py.File(path, 'r') as f:
#     ppg = np.array(f.get('/ppg')[:].astype(np.float32))
#     bp = np.array(f.get('/label')[:].astype(np.float32))
#
# # Use condition index(条件索引)
# filter_sbp = bp[(bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]                  # 952   # Separate filtering
# filter_dbp = filter_sbp[(filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]    # 881
#
# filter_sbp_ppg = ppg[(bp[:, sbp_index] >= lower_sbp_bound) & (bp[:, sbp_index] <= upper_sbp_bound)]             # 952  Separate filtering
# filter_dbp_ppg = filter_sbp_ppg[(filter_sbp[:, dbp_index] >= lower_dbp_bound) & (filter_sbp[:, dbp_index] <= upper_dbp_bound)]  # 881
# print()

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


# a = [x for x in range(80, 160, 10)]


# # 创建一个示例的 NumPy 数组
# A = np.array([102, 81, 98])
#
# # 计算标记的值
# labels = ((A - 80) // 10).astype(int)
#
# # 打印结果
# print(labels)
# print(labels[1])

# bp_hat = torch.rand(1, 17)
# sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
# sbp_label_hat, dbp_label_hat = bp_hat[:, 2:12], bp_hat[:, 12:]
# print()


# import torch
#
# # 一个假设的场景
# predictions = torch.randn((3, 4))  # 假设有3个样本，每个样本有4个预测
# targets = torch.randn((3, 2))  # 每个样本有2个回归目标
# class_labels = torch.randint(0, 4, (3,))  # 4个类别
#
# # 获取预测的类别
# predicted_classes = torch.argmax(predictions, dim=1)
#
# # 提取每个类别对应的回归目标
# a = predicted_classes.view(-1, 1)
# reg_targets_per_class = torch.gather(targets, 1, predicted_classes.view(-1, 1))

# import torch
#
# tensor_0 = torch.arange(3, 12).view(3, 3)
# print(tensor_0)
# index = torch.tensor([[2, 1, 0]])
# tensor_1 = tensor_0.gather(0, index)
# print(tensor_1)


# import torch
# import torch.nn.functional as F
#
#
# def decomposed_loss(predictions, targets, class_num):
#     """
#     将回归任务分解为分类任务和区间内回归任务的损失函数
#
#     参数：
#     - predictions: 模型的预测结果，一个包含分类分数和回归值的张量
#     - targets: 真实标签，包含分类标签和回归目标
#     - boundaries: 区间的边界列表
#
#     返回：
#     - 总体损失
#     """
#     # 分离分类标签和回归目标
#     class_targets, reg_targets = targets[:, 0].long(), targets[:, 1:].squeeze()
#
#     # 计算分类任务的损失（使用交叉熵损失）
#     classification_loss = F.cross_entropy(predictions[:, :class_num], class_targets)
#
#     # 获取预测的类别
#     a = F.softmax(predictions[:, :class_num])
#     # predicted_classes = torch.argmax(predictions[:, :len(boundaries)], dim=1)
#
#     # 计算区间内回归任务的损失（使用均方误差）
#     # a = predictions[:, -1]
#     # b = reg_targets
#
#     regression_loss = F.mse_loss(predictions[:, -1], reg_targets)
#
#     # 求和得到总体损失
#     total_loss = (classification_loss + regression_loss).mean()
#
#     return total_loss
#
#
# batch_size = 128
# # 示例用法
#
# # 假设有3个区间，每个区间对应一个类别
# num_classes = 3
#
#
# # 模型的预测结果，包括分类分数和回归值
# predictions = torch.randn((batch_size, num_classes + 1))
#
# # 真实标签，包括分类标签和回归目标
# # targets = torch.randn((batch_size, num_classes + 1))
# reg_targets = torch.randn((batch_size, 1))
# class_label = torch.randint(0, 3, (batch_size, 1))
# targets = torch.cat((class_label, reg_targets), dim=1)
#
# # 计算总体损失
# loss = decomposed_loss(predictions, targets, num_classes)
# print(loss.item())
#
# input = torch.randn(3, 5)
# target = torch.randint(5, (3,), dtype=torch.int64)
# loss = F.cross_entropy(input, target)
# print()


# import torch
#
# # 创建两个矩阵
# matrix1 = torch.tensor([[1, 2], [3, 4]])
# matrix2 = torch.tensor([[5, 6], [7, 8]])
#
# # 使用 torch.matmul 进行矩阵乘法
# result = torch.matmul(matrix2, matrix1)
#
# print("矩阵乘法的结果:")
# print(result)

