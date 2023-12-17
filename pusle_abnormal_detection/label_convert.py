# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/16 16:16
@Author  : Rao Zhi
@File    : label_convert.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('test_pulse_detection.csv')
df2 = pd.read_csv('train_pulse_detection.csv')

# 交换 "label" 列中的 0 和 1
df1['label'] = 1 - df1['label']
df2['label'] = 1 - df2['label']

# 将修改后的数据保存到新的CSV文件
df1.to_csv('modified_test_pulse_detection.csv', index=False)
df2.to_csv('modified_train_pulse_detection.csv', index=False)

