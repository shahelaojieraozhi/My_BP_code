# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/29 15:37
@Author  : Rao Zhi
@File    : bp_value_modify.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import numpy as np

new_bp = np.empty([110, 3])
bp_path = r'G:\Blood_Pressure_dataset\ours\subject_1\subject_bp.csv'
bp = np.loadtxt(bp_path, delimiter=',')
bp_ = bp.copy()
for i in range(55):
    new_bp[2 * i, :] = bp[i]
    new_bp[2 * i + 1, :] = bp_[i]




