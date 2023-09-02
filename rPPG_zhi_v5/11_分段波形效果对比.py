import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号"-"显示方框的问题


def normalization(data):
    mm_x = MinMaxScaler()
    data = mm_x.fit_transform(data)
    return data


def pulse_reshape(pulse_pd, seq_len):
    pulse_pd_ = pd.DataFrame(np.zeros([seq_len, 6]))

    # 把整个信号分成每段128帧
    for i in range(6):
        pulse_pd_[i] = pulse_pd.iloc[(i * seq_len):((i + 1) * seq_len), 0].reset_index(drop=True)
    reshaped_pulse = pd.DataFrame(normalization(pulse_pd_))
    return reshaped_pulse


if __name__ == '__main__':

    # param
    seq_len = 256

    ''' GT: shape = [-1, 1]'''
    Ground_truth = pd.read_csv('PPG_pulse.csv', header=None)
    Ground_truth = pd.DataFrame(normalization(Ground_truth))

    ''' GT: shape = [128, 6]'''
    reshape_GT_pulse = pulse_reshape(Ground_truth, seq_len=seq_len)

    ''' rPPG_pulse'''
    predict_pulse_file = './output/rPPG_signal.csv'
    predict_pulse_pd = pd.read_csv(predict_pulse_file, header=None)
    reshape_pulse = pulse_reshape(predict_pulse_pd, seq_len=seq_len)

    for i in range(6):
        plt.figure(i)
        reshape_pulse.iloc[:, i].plot(figsize=(12, 6))
        reshape_GT_pulse.iloc[:, i].plot(figsize=(12, 6))
        plt.legend(('predict', 'real'), loc='upper right', fontsize='15')
        plt.title("rPPG & PPG", fontsize='20')  # 添加标题
        plt.show()
