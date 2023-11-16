# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/19 10:05
@Author  : Rao Zhi
@File    : evaluate_personalize.py
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

    ax.set_title('Right ascension declination map')
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
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


if __name__ == '__main__':
    root_path = "finetune_result"
    for csv_path in os.listdir(root_path):
        bp = pd.read_csv(os.path.join(root_path, csv_path))

        # ['SBP_hat_pre', 'DBP_hat_pre', 'SBP_true', 'DBP_true', 'SBP_hat_post', 'DBP_hat_post']
        sbp_hat_pre = bp['SBP_hat_pre']
        dbp_hat_pre = bp['DBP_hat_pre']
        sbp_arr = bp['SBP_true']
        dbp_arr = bp['DBP_true']
        sbp_hat_post = bp['SBP_hat_post']
        dbp_hat_post = bp['DBP_hat_post']

        sbp_pre_sd, sbp_pre_mae, sbp_pre_rmse, sbp_pre_r_value = calculate_metrics(sbp_hat_pre, sbp_arr)
        dbp_pre_sd, dbp_pre_mae, dbp_pre_rmse, dbp_pre_r_value = calculate_metrics(dbp_hat_pre, dbp_arr)

        sbp_post_sd, sbp_post_mae, sbp_post_rmse, sbp_post_r_value = calculate_metrics(sbp_hat_post, sbp_arr)
        dbp_post_sd, dbp_post_mae, dbp_post_rmse, dbp_post_r_value = calculate_metrics(dbp_hat_post, dbp_arr)

        # plot_coordinates(sbp_arr, sbp_hat_arr, sbp_sd, sbp_mae)
        # plot_coordinates(dbp_arr, dbp_hat_arr, dbp_sd, dbp_mae, sbp=False)

        print(csv_path)
        # print("SBP Standard Deviation (SD) (pre - post):", sbp_pre_sd, sbp_post_sd)
        print("SBP Mean Absolute Error (MAE) (pre - post):", sbp_pre_mae, sbp_post_mae)
        # print("SBP Root Mean Square Error (RMSE) (pre - post):", sbp_rmse)
        # print("SBP Correlation Coefficient (r-value) (pre - post):", sbp_r_value)

        # print("DBP Standard Deviation (SD) (pre - post):", dbp_pre_sd, dbp_post_sd)
        print("DBP Mean Absolute Error (MAE) (pre - post):", dbp_pre_mae, dbp_post_mae)
        # print("DBP Root Mean Square Error (RMSE) (pre - post):", dbp_rmse)
        # print("DBP Correlation Coefficient (r-value) (pre - post):", dbp_r_value)
        print()
