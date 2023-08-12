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

import numpy as np
data = np.load('08-11-2023_pers_dataset.npz')
# a = data[0]
print(data['arr_0'].shape)  # (250000, 875)
print(data['arr_1'].shape)  # (250000, 2)
print(data['arr_2'].shape)  # (250000,)
