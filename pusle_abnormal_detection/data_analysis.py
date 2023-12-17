# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/16 18:07
@Author  : Rao Zhi
@File    : data_analysis.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import warnings
import torch

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ppgs = torch.load("ppg.h5")
labels = torch.load("label.h5")

# 计算 Tensor 中值为0的个数
a = torch.eq(labels, 0)
count_zeros = torch.sum(torch.eq(labels, 0).int())
count_ones = torch.sum(torch.eq(labels, 1).int())

print(f"Number of zero in the tensor: {count_zeros.item()}")
print(f"Number of one in the tensor: {count_ones.item()}")

# Number of zero in the tensor: 1201
# Number of one in the tensor: 300
