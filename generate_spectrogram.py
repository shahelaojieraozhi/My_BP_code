# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/28 19:31
@Author  : Rao Zhi
@File    : generate_spectrogram.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :
@detail   ：
@infer    :
"""
import numpy as np
import torch
from PIL import Image
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def main(ppg_signal, fs, mode, index):
    # 进行短时傅里叶变换
    frequencies, times, Sxx = spectrogram(ppg_signal, fs)
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.axis('off')

    # 将图像保存为.png
    plt.savefig(f'G:/Blood_Pressure_dataset/spectrogram/{mode}/{index}.png', bbox_inches='tight')

    # 关闭图形
    plt.close()

    # 读取保存的图像并调整大小
    img = Image.open(f'G:/Blood_Pressure_dataset/spectrogram/{mode}/{index}.png')
    resized_img = img.resize((224, 224))

    # 保存调整大小后的图像为.png
    resized_img.save(f'G:/Blood_Pressure_dataset/spectrogram/{mode}/{index}.png')


if __name__ == '__main__':
    mode = "test"
    fs = 125
    ppg = torch.load("pulse_abnormal_detection/dataset/" + mode + "/ppg.h5")
    # for idx, sample in tqdm(enumerate(ppg), desc="Processing process", unit="sample"):
    #     main(sample, fs, mode, idx)
    idx = 0
    for sample in tqdm(ppg, desc="Processing process", unit="sample"):
        main(sample, fs, mode, idx)
        idx += 1
