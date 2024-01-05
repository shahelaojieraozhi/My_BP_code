# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/13 14:13
@Author  : Rao Zhi
@File    : evaluate.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

warnings.filterwarnings("ignore")


def plot_coordinates(gt_bp, pre_bp, sd, mae, sbp=True):
    # plot_coordinates(gt_sbp, pre_sbp, sbp_sd, sbp_mae)

    # 创建一个新的图形
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # fig, ax = plt.subplots()
    if sbp:
        ax.set_xlim(-5, 500)
        ax.set_ylim(40, 200)
        ax.set_xticks(range(-5, 500, 50))
        ax.set_yticks(range(40, 200, 30))
    else:
        ax.set_xlim(-5, 500)
        ax.set_ylim(30, 120)
        ax.set_xticks(range(-5, 500, 50))
        ax.set_yticks(range(30, 120, 20))

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.scatter(np.arange(len(gt_bp)), gt_bp, color='k', s=50, label='gt')
    ax.scatter(np.arange(len(pre_bp)), pre_bp, color='r', s=50, label='pre')

    if sbp:
        ax.text(35, 80, 'sd: ' + str(int(sd)), fontsize=12, fontweight='bold', color='red', ha='center',
                va='center')

        ax.text(35, 60, 'mae: ' + str(int(mae)), fontsize=12, fontweight='bold', color='red', ha='center',
                va='center')

    else:
        ax.text(20, 110, 'sd: ' + str(int(sd)), fontsize=12, fontweight='bold', color='red', ha='center',
                va='center')

        ax.text(20, 115, 'mae: ' + str(int(mae)), fontsize=12, fontweight='bold', color='red', ha='center',
                va='center')

    ax.set_title('Blood Pressure Evaluation')
    ax.set_xlabel('subject index')
    ax.set_ylabel('bp value')
    ax.legend()
    plt.show()


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


def BHS_Standard(array1, array2):
    test_len = len(array1)
    count_a, count_b, count_c = 0, 0, 0
    for i in range(test_len):
        error = np.abs(array1[i] - array2[i])
        if error <= 5:
            count_a += 1
        if error <= 10:
            count_b += 1
        if error <= 15:
            count_c += 1

    return count_a / test_len, count_b / test_len, count_c / test_len


if __name__ == '__main__':

    """ This is my train result """
    test_path = '../predict_test/res_18_normal_val_loss_0.05076/'
    bps = []
    columns = ['sbp_hat_arr', 'dbp_hat_arr', 'sbp_arr', 'dbp_arr']
    for sec in os.listdir(test_path):
        single_sec = pd.read_csv(os.path.join(test_path, sec))
        bps.append(single_sec)

    bps = pd.DataFrame(np.concatenate(bps), columns=columns)

    sbp_hat_arr = bps['sbp_hat_arr']
    dbp_hat_arr = bps['dbp_hat_arr']
    sbp_arr = bps['sbp_arr']
    dbp_arr = bps['dbp_arr']

    sbp_sd, sbp_mae, sbp_rmse, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)
    dbp_sd, dbp_mae, dbp_rmse, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)

    # plot_coordinates(sbp_arr, sbp_hat_arr, sbp_sd, sbp_mae)
    # plot_coordinates(dbp_arr, dbp_hat_arr, dbp_sd, dbp_mae, sbp=False)

    print()
    print("SBP Standard Deviation (SD):", sbp_sd)
    print("SBP Mean Absolute Error (MAE):", sbp_mae)
    print("SBP Root Mean Square Error (RMSE):", sbp_rmse)
    print("SBP Correlation Coefficient (r-value):", sbp_r_value)

    print("DBP Standard Deviation (SD):", dbp_sd)
    print("DBP Mean Absolute Error (MAE):", dbp_mae)
    print("DBP Root Mean Square Error (RMSE):", dbp_rmse)
    print("DBP Correlation Coefficient (r-value):", dbp_r_value)
    print()

    # bp = pd.read_csv('./resnet_ppg_nonmixed_test_results.csv')
    #
    # sbp_hat_arr = bp['SBP_est']
    # dbp_hat_arr = bp['DBP_est']
    # sbp_arr = bp['SBP_true']
    # dbp_arr = bp['DBP_true']
    #
    # sbp_sd, sbp_mae, sbp_rmse, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)
    # dbp_sd, dbp_mae, dbp_rmse, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)
    #
    # # plot_coordinates(sbp_arr, sbp_hat_arr, sbp_sd, sbp_mae)
    # # plot_coordinates(dbp_arr, dbp_hat_arr, dbp_sd, dbp_mae, sbp=False)
    #
    # print()
    # print("SBP Standard Deviation (SD):", sbp_sd)
    # print("SBP Mean Absolute Error (MAE):", sbp_mae)
    # print("SBP Root Mean Square Error (RMSE):", sbp_rmse)
    # print("SBP Correlation Coefficient (r-value):", sbp_r_value)
    #
    # print("DBP Standard Deviation (SD):", dbp_sd)
    # print("DBP Mean Absolute Error (MAE):", dbp_mae)
    # print("DBP Root Mean Square Error (RMSE):", dbp_rmse)
    # print("DBP Correlation Coefficient (r-value):", dbp_r_value)
    # print()
