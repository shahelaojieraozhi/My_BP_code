# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/9/2 9:29
@Author  : Rao Zhi
@File    : read_mp150_pusle.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


path = r'sub_6.txt'
# 打开文件
f = open(path, encoding='utf-8')
# 创建空列表
text = []
# 读取全部内容 ，并以列表方式返回
lines = f.readlines()
container = np.ones([len(lines), 2])
i = 0
for line in lines:
    # 如果读到空行，就跳过
    if line.isspace():
        continue
    else:
        # 去除文本中的换行等等，可以追加其他操作
        line = line.replace("\n", "")
        line = line.split('\t')
        # 处理完成后的行，追加到列表中
        container[i, 0] = float(line[0])
        container[i, 1] = float(line[1])

    i += 1

# pd.DataFrame(container).to_csv('sub_6.csv', header=False, index=False)

time = float(len(container) / 1000)
ECG_resampled = signal.resample(container[:, 0], int(100 * time))
PPG_resampled = signal.resample(container[:, 1], int(30 * time))
pd.DataFrame(PPG_resampled).to_csv('PPG_pulse_sub_6.csv', header=False, index=False)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(ECG_resampled)
# plt.title("原始信号 (1000 Hz采样)")
plt.subplot(2, 1, 2)
plt.plot(PPG_resampled)
# plt.title("降采样后的信号 (30 Hz采样)")
plt.show()

