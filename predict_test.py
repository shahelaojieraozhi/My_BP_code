# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/11 14:45
@Author  : Rao Zhi
@File    : predict_test.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.resnet18_1D import resnet18_1d
# from PPG2BP_Dataset_sbp_dbp import PPG2BPDataset
from model.Resnet import resnet50, resnet34, resnet18, resnet101, resnet152
from model.MSR_tranformer_bp import msr_tf_bp

from PPG2BP_Dataset import PPG2BPDataset, use_derivative
from model.bpnet_cvprw import resnet50
import os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_calculate(pre, label, loss_name='mse'):
    if loss_name == 'mse':
        return F.mse_loss(pre, label)
    elif loss_name == "SmoothL1Loss":
        smooth_l1_loss = torch.nn.SmoothL1Loss()
        return smooth_l1_loss(pre, label)


def inv_normalize(sbp_arr, dbp_arr):
    sbp_min = 40
    sbp_max = 200
    dbp_min = 40
    dbp_max = 120
    sbp_arr = sbp_arr * (sbp_max - sbp_min) + sbp_min
    dbp_arr = dbp_arr * (dbp_max - dbp_min) + dbp_min
    return sbp_arr, dbp_arr


def test(model, opt, pre_path):
    print('loading data...')
    test_data = PPG2BPDataset('test')
    test_loader = DataLoader(test_data, batch_size=opt.batch, num_workers=0)
    model.eval()

    loss_meter, it_count = 0, 0
    test_batch_idx = 0
    with torch.no_grad():
        for (ppg, sbp, dbp) in test_loader:
            ppg = use_derivative(ppg) if opt.using_derivative else ppg
            ppg = ppg.to(device)
            bp_hat = model(ppg).cpu()
            sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
            # dbp_hat, sbp_hat = bp_hat[:, 0], bp_hat[:, 1]  # error

            sbp_hat_arr = sbp_hat.numpy()
            dbp_hat_arr = dbp_hat.numpy()

            sbp_arr = sbp.numpy()
            dbp_arr = dbp.numpy()

            # sbp_arr, dbp_arr = inv_normalize(sbp_arr, dbp_arr)
            # sbp_hat_arr, dbp_hat_arr = inv_normalize(sbp_hat_arr, dbp_hat_arr)

            table_arr = np.vstack((sbp_hat_arr, dbp_hat_arr, sbp_arr, dbp_arr)).T

            pd.DataFrame(table_arr).to_csv(pre_path + "/predict_test_{}.csv".format(test_batch_idx),
                                           header=['sbp_hat_arr', 'dbp_hat_arr', 'sbp_arr', 'dbp_arr'], index=False)
            print("predict_test_{}.csv is written".format(test_batch_idx))

            loss_sbp = loss_calculate(sbp_hat, sbp, opt.loss_func)
            loss_dbp = loss_calculate(dbp_hat, dbp, opt.loss_func)

            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            it_count += 1
            test_batch_idx += 1

    return loss_meter / it_count


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


def evaluate(test_path):
    """ This is my train result """
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
    print()

    print("DBP Standard Deviation (SD):", dbp_sd)
    print("DBP Mean Absolute Error (MAE):", dbp_mae)
    print("DBP Root Mean Square Error (RMSE):", dbp_rmse)
    print("DBP Correlation Coefficient (r-value):", dbp_r_value)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=4096, help="batch size of training")
    # logs/11_9_add_wd/best_w.pth
    # parser.add_argument("-m", "--model_name", type=str, default='cvpr_no_decay', help="model name")  # best
    # parser.add_argument("-m", "--model_name", type=str, default='msr_tf_bp_normal_bp_2023111308', help="model to execute")
    parser.add_argument("-m", "--model_name", type=str, default='cvprw_reproduce_new_data', help="model to execute")    # vs cvprw
    parser.add_argument('--using_derivative', default=False, help='using derivative of PPG or not')
    parser.add_argument('--loss_func', type=str, default='SmoothL1Loss', help='which loss function is selected')
    # parser.add_argument("-tp", "--test_data_path", type=str,
    #                     default='G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test', help="test data path")
    opt = parser.parse_args()

    input_channel = 3 if opt.using_derivative else 1
    "model"
    # resnet_1d = resnet18_1d()
    # resnet_1d = resnet50()
    # model = resnet_1d.to(device)
    # model = resnet50(num_input_channels=1, num_classes=2)
    # model = MSResNet(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)
    # model = resnet18()
    # model = resnet18(input_c=1 if input_channel == 1 else 3, num_classes=2)
    # model = model.to(device)

    model = resnet50(input_c=input_channel, num_classes=2)   # cvprw

    # model = msr_tf_bp(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)    # ours
    model = model.to(device)

    "load model"
    model.load_state_dict(torch.load('logs/' + opt.model_name + '/best_w.pth')['state_dict'])
    best_epoch = torch.load('logs/' + opt.model_name + '/best_w.pth')["epoch"]
    pre_path = os.path.join("predict_test", opt.model_name)
    os.makedirs(pre_path, exist_ok=True)
    test_loss = test(model, opt, pre_path)

    print("The epoch number of the best result:", best_epoch)
    evaluate(pre_path)
    print(test_loss)

# tensorboard --logdir=resnet18_202307141720 --port=6007
# tensorboard --logdir=add_normal_res_18 --port=6007
