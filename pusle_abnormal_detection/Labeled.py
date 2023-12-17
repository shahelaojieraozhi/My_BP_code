# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/15 10:45
@Author  : Rao Zhi
@File    : Labeled.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import matplotlib.pyplot as plt

# mode = 'train'
mode = 'test'
ppg = torch.load("../data_normal/" + mode + "/ppg.h5")

# batch show

internal = 8
grouped_waveforms = [ppg[i:i + internal] for i in range(0, len(ppg), internal)]

for group_idx, grouped_waveform in enumerate(grouped_waveforms):
    column = [str(group_idx * internal + i) for i in range(internal)]
    fig = plt.figure(figsize=(16, 6))
    for i in range(internal):
        plt.subplot(int(internal/4), 4, i + 1)
        plt.plot(grouped_waveform[i])
        plt.title(column[i], fontsize=10)
    plt.show()
