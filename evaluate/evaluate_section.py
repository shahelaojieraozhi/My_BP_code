# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/20 11:46
@Author  : Rao Zhi
@File    : evaluate_section.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings("ignore")


def calculate_metrics(array1, array2):
    # Standard Deviation (SD)
    sd = np.std(array1 - array2)

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(array1 - array2))

    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((array1 - array2) ** 2))

    # Correlation Coefficient (r-value)
    r_value, _ = pearsonr(array1, array2)

    return sd, mae, rmse, r_value


def plot_coordinates(gt_bp, pre_bp):
    # 创建一个新的图形
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.scatter(np.arange(len(gt_bp)), gt_bp, color='k', s=50, label='gt')
    ax.scatter(np.arange(len(pre_bp)), pre_bp, color='r', s=50, label='pre')

    ax.set_title('bp pre-gt map')
    ax.set_xlabel('subject section')
    ax.set_ylabel('bp value')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    """ This is my train result """
    # test_path = '../predict_test/msr_tf_bp_normal_bp_2023111308/'
    test_path = '../predict_test/normal_bp_msr_tf_bp_bs=16384_HuberLoss_no-fixed_lr_2023111706/'  # cvprw result
    bps = []
    columns = ['sbp_hat_arr', 'dbp_hat_arr', 'sbp_arr', 'dbp_arr']
    for sec in os.listdir(test_path):
        single_sec = pd.read_csv(os.path.join(test_path, sec))
        bps.append(single_sec)

    bps = pd.DataFrame(np.concatenate(bps), columns=columns)

    """ This is the paper result """
    # bps = pd.read_csv("./resnet_ppg_nonmixed_test_results.csv")

    sbp_section = [x for x in range(80, 160, 10)]
    for threshold in sbp_section:
        sbp_split = bps[(threshold < bps['sbp_arr']) & (bps['sbp_arr'] < int(threshold) + 10)]
        sbp_hat_arr = sbp_split['sbp_hat_arr']
        sbp_arr = sbp_split['sbp_arr']
        sbp_sd, sbp_mae, _, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)

        print()
        print("SBP section is {} ~ {}".format(threshold, threshold + 10))
        print("SBP section number is {}".format(len(sbp_arr)))
        # print("SBP Standard Deviation (SD):", sbp_sd)
        print("SBP Mean Absolute Error (MAE):", sbp_mae)
        # print("SBP Correlation Coefficient (r-value):", sbp_r_value)
        # print()
        # plot_coordinates(sbp_arr, sbp_hat_arr)

    dbp_section = [x for x in range(50, 80, 10)]

    for threshold in dbp_section:
        dbp_split = bps[(threshold < bps['dbp_arr']) & (bps['dbp_arr'] < int(threshold) + 10)]
        dbp_hat_arr = dbp_split['dbp_hat_arr']
        dbp_arr = dbp_split['dbp_arr']

        dbp_sd, dbp_mae, _, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)

        print()
        print("DBP section is {} ~ {}".format(threshold, threshold + 10))
        print("DBP section number is {}".format(len(dbp_arr)))
        # print("DBP Standard Deviation (SD):", dbp_sd)
        print("DBP Mean Absolute Error (MAE):", dbp_mae)
        # print("DBP Correlation Coefficient (r-value):", dbp_r_value)
        # print()
        # plot_coordinates(dbp_arr, dbp_hat_arr)
