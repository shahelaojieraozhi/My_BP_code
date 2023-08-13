# -*- coding: utf-8 -*-
"""
@Project ：My_bp_code 
@Time    : 2023/8/12 17:02
@Author  : Rao Zhi
@File    : personalization_training.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import os
import shutil
import time
import argparse

import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import utils
from resnet18_1D import resnet18_1d, resnet34_1d
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# from PPG2BP_Dataset_v2 import PPG2BPDataset
# from Transformer_reg import TF
from Transformer_reg_v2 import RegressionTransformer

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, train_dataloader, show_interval=10):
    model.train()
    loss_meter, it_count = 0, 0

    for (ppg, sbp, dbp) in train_dataloader:
        # tf
        ppg = ppg.squeeze(1)

        # other
        ppg = ppg.to(device)
        bp_hat = model(ppg).cpu()

        sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
        optimizer.zero_grad()

        loss_sbp = F.mse_loss(sbp_hat, sbp)
        loss_dbp = F.mse_loss(dbp_hat, dbp)
        loss = loss_dbp + loss_sbp

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()

        it_count += 1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d, loss: %.3e" % (it_count, loss_meter))  # show the sum loss of every show_interval

    return loss_meter


def val_epoch(model, optimizer, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    with torch.no_grad():
        for (ppg, sbp, dbp) in val_dataloader:
            # transformer
            ppg = ppg.squeeze(1)
            # other
            ppg = ppg.to(device)
            bp_hat = model(ppg).cpu()
            sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
            optimizer.zero_grad()

            loss_sbp = F.mse_loss(sbp_hat, sbp)
            loss_dbp = F.mse_loss(dbp_hat, dbp)
            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            it_count += 1

    return loss_meter


def inv_normalize(sbp_arr, dbp_arr):
    sbp_min = 40
    sbp_max = 200
    dbp_min = 40
    dbp_max = 120

    sbp_arr = sbp_arr * (sbp_max - sbp_min) + sbp_min
    dbp_arr = dbp_arr * (dbp_max - dbp_min) + dbp_min

    return sbp_arr, dbp_arr


def train():
    print('loading data...')

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=2048, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='resnet18', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    parser.add_argument('--N_trials', type=int, default=20,
                        help="Number subjects used for personalization (default :20)")
    opt = parser.parse_args()

    "model"
    # model = TF(in_features=875, drop=0.).to(device)
    # model = RegressionTransformer(input_dim=875, output_dim=2)
    # model = resnet34_1d().to(device)
    # resnet_1d = resnet50()
    # model = resnet_1d.to(device)

    resnet_1d = resnet18_1d()
    # resnet_1d = resnet50()
    model = resnet_1d.to(device)

    model.load_state_dict(torch.load('save/resnet18_202307141720/best_w.pth')['state_dict'])  # 50

    # for name, param in model.named_parameters():
    #     # print(name)
    #     # print(param)
    #     print(name, "   ", param.shape)
    #
    # a1 = list(model.parameters())
    # a = list(model.parameters())[:-8]

    for param in list(model.parameters())[:-8]:
        param.requires_grad = False

    # model_save_dir = f'save/{opt.type}_{time.strftime("%Y%m%d%H%M")}'
    # os.makedirs(model_save_dir, exist_ok=True)

    # test_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\test"
    # test_data = PPG2bpDataset(test_data_path)
    # test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    pd_col_names = ['subject', 'Sbp_true', 'Dbp_true', 'Sbp_est_prepers', 'Dbp_est_prepers', 'Sbp_est_postpers',
                    'Dbp_est_postpers']
    results = pd.DataFrame([], columns=pd_col_names)

    # load the test dataset of MIMIC-iii (here, it is regarded as the train dataset)
    data = np.load('08-11-2023_pers_dataset.npz')

    ppg = data['arr_0']  # shape = (250000, 875)
    bp = data['arr_1']  # shape = (250000, 875)
    subject_idx = data['arr_2']  # shape = (250000, 875)
    subjects = np.unique(subject_idx)

    trial_subjects = np.random.choice(subjects, size=opt.N_trials, replace=False)
    # N_trials  :  Number subjects used for personalization (default :20)
    with open('ppg_personalization_subject_list.txt', 'w') as f:
        for item in trial_subjects:
            f.write(("%s\n" % item))

    # perform personalization for each test subject
    for subject in trial_subjects:
        print(f'Processing subject {subject} of {len(trial_subjects)}')

        ppg_trial = ppg[subject_idx == subject, :]
        bp_trial = bp[subject_idx == subject, :]
        n_same_subject_idx = bp_trial.shape[0]
        n_train = int(np.round(0.2 * n_same_subject_idx))
        # get 20% data of a subject(subject_idx == subject) as train data

        idx_test = np.arange(n_train + 1, n_same_subject_idx, 2)  # why?
        ppg_test = ppg_trial[idx_test, :]  # (87, 875)
        bp_test = bp_trial[idx_test, :]  # (87, 2)

        ppg_trial = np.delete(ppg_trial, idx_test, axis=0)
        bp_trial = np.delete(bp_trial, idx_test, axis=0)

        random_pick = True

        # draw training data from the test subject's data
        if random_pick:
            idx_train, idx_val = train_test_split(range(ppg_trial.shape[0]), test_size=int(n_train), shuffle=True)
            ppg_train = ppg_trial[idx_train, :]  # ndarray (87, 875)
            bp_train = bp_trial[idx_train, :]  # ndarray (87, 2)
            ppg_val = ppg_trial[idx_val, :]  # ndarray (43, 875)
            bp_val = bp_trial[idx_val, :]  # ndarray (43, 2)
        else:
            ppg_train = ppg_trial[:n_train, :]
            bp_train = bp_trial[:n_train, :]
            ppg_val = ppg_trial[:n_train, :]
            bp_val = bp_trial[:n_train, :]

        # to tensor

        def ndarray2tensor(x):
            return torch.from_numpy(x.astype(np.float32))

        ppg_test_tensor = ndarray2tensor(ppg_test).unsqueeze(dim=0)  # (87, 875)
        # bp_test_tensor = ndarray2tensor(bp_test)  # (87, 2)
        ppg_train_tensor = ndarray2tensor(ppg_train).unsqueeze(dim=0)  # (87, 875)
        # bp_train_tensor = ndarray2tensor(bp_train)  # (87, 2)
        ppg_val_tensor = ndarray2tensor(ppg_val).unsqueeze(dim=0)  # (87, 875)
        # bp_val_tensor = ndarray2tensor(bp_val)  # (87, 2)

        ppg_test_tensor = torch.transpose(ppg_test_tensor, 1, 0)
        ppg_train_tensor = torch.transpose(ppg_train_tensor, 1, 0)
        ppg_val_tensor = torch.transpose(ppg_val_tensor, 1, 0)

        torch_container = torch.empty([2048, 1, 875])
        torch_container[:len(ppg_test_tensor), :, :] = ppg_test_tensor
        torch_container = torch_container.to(device)
        bp_val_pre_pers_hat = model(torch_container).cpu()

        bp_val_pre_pers_hat_arr = bp_val_pre_pers_hat.numpy()

        # sbp_val_pre_pers, dbp_val_pre_pers

        sbp_arr, dbp_arr = inv_normalize(sbp_arr, dbp_arr)
        sbp_hat_arr, dbp_hat_arr = inv_normalize(sbp_hat_arr, dbp_hat_arr)

        # SBP_train = BP_train[:, 0]
        # DBP_train = BP_train[:, 1]
        # SBP_val = BP_val[:, 0]
        # DBP_val = BP_val[:, 1]

        # best_lost = 1e3
        # lr = 1e-4
        # start_epoch = 1
        # stage = 1
        # step = [15, 25]
        # weight_decay = 2
        #
        # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #
        # states = []
        #
        # for epoch in range(start_epoch, opt.n_epochs):
        #     since = time.time()
        #     train_loss = train_epoch(model, optimizer, train_loader, 50)
        #     val_loss = val_epoch(model, optimizer, val_loader)
        #
        #     print('#epoch: %02d stage: %d train_loss: %.3e val_loss: %0.3e time: %s\n'
        #           % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)))
        #
        #     writer = SummaryWriter(model_save_dir)
        #     writer.add_scalar('train_loss', train_loss, epoch)  # add_scalar 添加标量
        #     writer.add_scalar('val_loss', val_loss, epoch)  # add_scalar 添加标量
        #     writer.close()
        #
        #     state = {"state_dict": model.state_dict(), "epoch": epoch,
        #              "loss": val_loss, 'lr': lr, 'stage': stage}
        #
        #     states.append(state)
        #
        #     save_ckpt(state, best_lost > val_loss, model_save_dir)
        #     best_lost = min(best_lost, val_loss)
        #
        #     if epoch in step:
        #         stage += 1
        #         lr /= 10
        #
        #         print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
        #         utils.adjust_learning_rate(optimizer, lr)
        #
        # torch.save(states, f'./save/resnet18_1D_states.pth')


if __name__ == '__main__':
    train()

# tensorboard --logdir=cnn_202305061217 --port=6007
