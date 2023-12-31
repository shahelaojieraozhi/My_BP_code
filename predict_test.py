# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/11 14:45
@Author  : Rao Zhi
@File    : evaluate-reproduce_result.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from model.resnet18_1D import resnet18_1d
# from PPG2BP_Dataset_sbp_dbp import PPG2BPDataset
# from model.Resnet import resnet50, resnet34, resnet18, resnet101, resnet152

# compare with ourselves
from model.MSR_tranformer_bp import msr_tf_bp  # msr_tf_bp_mse_data_split+derivative_2023121802
# from model.MK_ResNet import MK_ResNet  # msr_tf_bp_mse_data_split+derivative_2023121802
from model.ResNet_Transformer import resnet50  # ResNet_Transformer

# from model.MSR_tranformer_bp_v10 import msr_tf_bp
# from model.MSR_tranformer_bp_v2 import msr_tf_bp
# from model.MSR_tranformer_bp_v7 import msr_tf_bp      #
# from model.MSR_tranformer_bp_v7 import msr_tf_bp  # only one se module (using dbp_best_w.pth)

# from PPG2BP_Dataset import PPG2BPDataset, use_derivative
from PPG2BP_Dataset_filter_pulse import PPG2BPDataset, use_derivative

# from model.bpnet_cvprw import resnet50
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
    elif loss_name == "HuberLoss":
        HuberLoss = torch.nn.HuberLoss(delta=1)
        return HuberLoss(pre, label)


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
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(array1 - array2))

    # Standard Deviation
    std = np.sqrt(np.mean((abs(array1 - array2) - mae) ** 2))
    # sd = np.std(array1 - array2)

    # Correlation Coefficient (r-value)
    r_value, _ = pearsonr(array1, array2)

    return mae, std, r_value


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


def bp_classifier_by_radius(array_hat, array_gt):
    desired_indices_gt = np.where(array_gt <= 120)
    desired_indices_hat = np.where(array_hat <= 120)
    desired_bp_acc = len(np.intersect1d(desired_indices_gt, desired_indices_hat)) / len(desired_indices_gt[0])

    pre_indices_gt = np.where((120 < array_gt) & (array_gt <= 140))
    pre_indices_hat = np.where((120 < array_hat) & (array_hat <= 140))
    pre_bp_acc = len(np.intersect1d(pre_indices_gt, pre_indices_hat)) / len(pre_indices_gt[0])

    hy_indices_gt = np.where(140 < array_gt)
    hy_indices_hat = np.where(140 < array_hat)
    hy_bp_acc = len(np.intersect1d(hy_indices_gt, hy_indices_hat)) / len(hy_indices_gt[0])

    return desired_bp_acc, pre_bp_acc, hy_bp_acc


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

    map_arr = bps['dbp_arr'] + (bps['sbp_arr'] - bps['dbp_arr']) / 3
    map_hat_arr = bps['dbp_hat_arr'] + (bps['sbp_hat_arr'] - bps['dbp_hat_arr']) / 3

    # desired_bp_acc, pre_bp_acc, hy_bp_acc = bp_classifier_by_radius(sbp_hat_arr, sbp_arr)

    # mae, std, rmse, r_value
    sbp_mae, sbp_std, sbp_r_value = calculate_metrics(sbp_hat_arr, sbp_arr)
    dbp_mae, dbp_std, dbp_r_value = calculate_metrics(dbp_hat_arr, dbp_arr)
    map_mae, map_std, map_r_value = calculate_metrics(map_hat_arr, map_arr)

    sbp_a_percen, sbp_b_percen, sbp_c_percen = BHS_Standard(sbp_hat_arr, sbp_arr)
    dbp_a_percen, dbp_b_percen, dbp_c_percen = BHS_Standard(dbp_hat_arr, dbp_arr)
    map_a_percen, map_b_percen, map_c_percen = BHS_Standard(map_hat_arr, map_arr)

    # plot_coordinates(sbp_arr, sbp_hat_arr, sbp_sd, sbp_mae)
    # plot_coordinates(dbp_arr, dbp_hat_arr, dbp_sd, dbp_mae, sbp=False)

    # print("SBP Mean Absolute Error (MAE):", sbp_mae)
    # print("DBP Mean Absolute Error (MAE):", dbp_mae)

    print()
    print("SBP Mean Absolute Error (MAE):", sbp_mae)
    print("SBP Standard Deviation (STD):", sbp_std)
    print("SBP Correlation Coefficient (r-value):", sbp_r_value)
    print()

    print("DBP Mean Absolute Error (MAE):", dbp_mae)
    print("DBP Standard Deviation (STD):", dbp_std)
    print("DBP Correlation Coefficient (r-value):", dbp_r_value)
    print()

    print("MAP Mean Absolute Error (MAE):", map_mae)
    print("MAP Standard Deviation (STD):", map_std)
    print("MAP Correlation Coefficient (r-value):", map_r_value)
    print()

    result_metrics = [sbp_mae, sbp_std, sbp_r_value, dbp_mae, dbp_std, dbp_r_value, map_mae, map_std, map_r_value]

    BHS_Standards = [[sbp_a_percen, sbp_b_percen, sbp_c_percen]]
    BHS_Standards.append([map_a_percen, map_b_percen, map_c_percen])
    BHS_Standards.append([dbp_a_percen, dbp_b_percen, dbp_c_percen])

    return result_metrics, BHS_Standards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=4096, help="batch size of training")
    parser.add_argument("-mn", "--model_name", type=str,
                        # default='best_filter_pulse',  # un
                        # default='cvpr_no',  # reproduce result

                        # new_data (normal bp value dataset)
                        # default='cvprw_reproduce_new_data',     # 75 ~ 165 mmHg & 40 ~ 80 mmHg (single channel)
                        # default='cvpr_normal_data_3_channel',     # 75 ~ 165 mmHg & 40 ~ 80 mmHg (3 channel)

                        # split_date
                        # default='cvprw_split_data__2023121909',   # new baseline (3 channel)
                        # default='cvprw_split_data_1_channel',  # new baseline (3 channel)

                        # default='msr_tf_bp_best_dbp_autodl',
                        # default='msr_tf_bp_mse_data_split+derivative_2023121802',
                        # default='msr_tf_bp_SmoothL1Loss_data_split_2023121709',           # MAE = 10.8591031145226
                        # default='msr_tf_bp_SmoothL1Loss_data_split+derivative_2023121801',  # MAE = 10.372140880383546

                        # default='MK-Resnet_3 channel',  # MAE = 10.372140880383546
                        default='ResNet_Transformer_3_channel',  # MAE = 10.372140880383546

                        # default='msr_tf_bp_best_dbp_autodl',
                        # default='msr_tf_bp_se__2023122503',  # best;  using dbp_best_w.pth
                        # default='msr_tf_bp-se_before pooling_2023122700',  # best
                        help="model to execute")  # vs cvprw
    parser.add_argument("-m", "--model", type=str, default='ResNet_Transformer',
                        choices=('msr_tf_bp', 'cvprw', 'MK_ResNet', 'ResNet_Transformer'),
                        help="model to execute")
    parser.add_argument('--using_derivative', default=True, help='using derivative of PPG or not')
    parser.add_argument('--loss_func', type=str, default='SmoothL1Loss',
                        choices=('SmoothL1Loss', 'mse', 'bp_bucketing_loss', 'HuberLoss'),
                        help='which loss function is selected')
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

    if opt.model == 'cvprw':
        model = resnet50(input_c=input_channel, num_classes=2)  # cvprw
    elif opt.model == 'msr_tf_bp':
        model = msr_tf_bp(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)  # ours
    elif opt.model == "MK_ResNet":
        model = MK_ResNet(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)  # ours
    elif opt.model == "ResNet_Transformer":
        model = resnet50(input_c=3, num_classes=2)  # ours
    else:
        pass
    model = model.to(device)

    pre_path = os.path.join("predict_test", opt.model_name)
    if os.path.exists(pre_path):

        metrics, BHS_Standards = evaluate(pre_path)
        # metrics.append(test_loss)
        # print("Test loss:", test_loss)
        pd.DataFrame(metrics).T.to_csv("evaluate/metric_result.csv", header=False, index=False)
        pd.DataFrame(np.array(BHS_Standards)).to_csv("evaluate/BHS_Standards.csv", header=False,
                                                     index=False)

    else:

        "load model"
        # model.load_state_dict(torch.load('logs/' + opt.model_name + '/dbp_best_w.pth')['state_dict'])
        model.load_state_dict(torch.load('logs/' + opt.model_name + '/best_w.pth')['state_dict'])
        best_epoch = torch.load('logs/' + opt.model_name + '/best_w.pth')["epoch"]

        os.makedirs(pre_path, exist_ok=True)
        test_loss = test(model, opt, pre_path)

        print()
        # print model name
        describe = ""
        print(opt.model_name)
        print(f"best epoch: {best_epoch}")
        print(describe)

        metrics, BHS_Standards = evaluate(pre_path)
        metrics.append(test_loss)
        print("Test loss:", test_loss)
        pd.DataFrame(metrics).T.to_csv("evaluate/metric_result.csv", header=False, index=False)
        pd.DataFrame(np.array(BHS_Standards)).to_csv("evaluate/BHS_Standards.csv", header=False,
                                                     index=False)

# tensorboard --logdir=resnet18_202307141720 --port=6007
# tensorboard --logdir=add_normal_res_18 --port=6007
