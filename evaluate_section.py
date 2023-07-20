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
    bp = pd.read_csv("predict_test/18_60_10e4/predict_test_1.csv")
    sbp_section_list = [x for x in range(60, 200, 20)]
    for threshold in sbp_section_list:
        sbp_split = bp[(threshold < bp['sbp_arr']) & (bp['sbp_arr'] < int(threshold) + 10)]
        sbp_hat_arr = sbp_split['sbp_hat_arr']
        sbp_arr = sbp_split['sbp_arr']
        sbp_sd, sbp_mae, _, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)

        print()
        print("SBP section is {} ~ {}".format(threshold, threshold + 10))
        print("SBP section number is {}".format(len(sbp_arr)))
        print("SBP Standard Deviation (SD):", sbp_sd)
        print("SBP Mean Absolute Error (MAE):", sbp_mae)
        print()
        plot_coordinates(sbp_arr, sbp_hat_arr)

    # dbp_section_list = [x for x in range(40, 120, 20)]
    #
    # for threshold in dbp_section_list:
    #     dbp_split = bp[(threshold < bp['dbp_arr']) & (bp['dbp_arr'] < int(threshold) + 10)]
    #     dbp_hat_arr = dbp_split['dbp_hat_arr']
    #     dbp_arr = dbp_split['dbp_arr']
    #
    #     dbp_sd, dbp_mae, _, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)
    #
    #     print()
    #     print("DBP section is {} ~ {}".format(threshold, threshold + 10))
    #     print("DBP section number is {}".format(len(dbp_arr)))
    #     print("DBP Standard Deviation (SD):", dbp_sd)
    #     print("DBP Mean Absolute Error (MAE):", dbp_mae)
    #     print()
    #     plot_coordinates(dbp_arr, dbp_hat_arr)

