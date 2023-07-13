# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/13 14:13
@Author  : Rao Zhi
@File    : evaluate.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


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
    bp = pd.read_csv("predict_test/res_normal_18/predict_test_4.csv")

    sbp_hat_arr = bp['sbp_hat_arr']
    dbp_hat_arr = bp['dbp_hat_arr']
    sbp_arr = bp['sbp_arr']
    dbp_arr = bp['dbp_arr']

    sbp_sd, sbp_mae, sbp_rmse, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)
    dbp_sd, dbp_mae, dbp_rmse, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)

    print("SBP Standard Deviation (SD):", sbp_sd)
    print("SBP Mean Absolute Error (MAE):", sbp_mae)
    print("SBP Root Mean Square Error (RMSE):", sbp_rmse)
    print("SBP Correlation Coefficient (r-value):", sbp_r_value)

    print("DBP Standard Deviation (SD):", dbp_sd)
    print("DBP Mean Absolute Error (MAE):", dbp_mae)
    print("DBP Root Mean Square Error (RMSE):", dbp_rmse)
    print("DBP Correlation Coefficient (r-value):", dbp_r_value)
