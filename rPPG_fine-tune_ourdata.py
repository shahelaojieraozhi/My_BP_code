# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code
@Time    : 2023/8/20 9:47
@Author  : Rao Zhi
@File    : rPPG_fine-tune.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm

"""

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import utils
from model.resnet18_1D import resnet18_1d
from PPG2BP_Dataset_finetune import TDataset

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed=42)


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def del_file(path_data):
    """delete all directory in the current path"""
    for element in os.listdir(path_data):
        file_data = path_data + "\\" + element
        if os.path.isfile(file_data):
            os.remove(file_data)
        else:
            del_file(file_data)


def train_epoch(model, optimizer, train_dataloader):
    model.train()
    loss_meter, it_count = 0, 0

    for (ppg, sbp, dbp) in train_dataloader:
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

        # it_count += 1
        # if it_count != 0 and it_count % show_interval == 0:
        #     print("%d, loss: %.3e" % (it_count, loss_meter))  # show the sum loss of every show_interval

    return loss_meter


def val_epoch(model, optimizer, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    with torch.no_grad():
        for (ppg, sbp, dbp) in val_dataloader:
            # transformer
            # ppg = ppg.squeeze(1)
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


def bp_predict(model, test_loader):
    test_batch_idx = 0
    bp_list = []
    with torch.no_grad():
        for (ppg, sbp, dbp) in test_loader:
            ppg = ppg.to(device)
            bp_hat = model(ppg).cpu()
            sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]

            dbp_hat_arr = dbp_hat.numpy()
            sbp_hat_arr = sbp_hat.numpy()

            sbp_arr = sbp.numpy()
            dbp_arr = dbp.numpy()

            sbp_hat_arr, dbp_hat_arr = inv_normalize(sbp_hat_arr, dbp_hat_arr)
            bp_arr = np.vstack((sbp_hat_arr, dbp_hat_arr, sbp_arr, dbp_arr)).T
            bp_list.append(bp_arr)
            test_batch_idx += 1
    return np.concatenate(bp_list, axis=0)


def fine_tuning(opt):
    # model
    resnet_1d = resnet18_1d()
    model = resnet_1d.to(device)

    # load rPPG data from the provided hdf5 files
    with h5py.File(opt.data_file, 'r') as f:
        rppg = f.get('rPPG')
        BP = f.get('label')
        subjects = f.get('subject_idx')

        rppg = np.transpose(np.array(rppg))
        BP = np.transpose(np.array(BP))  # (7851, 2)
        subjects = np.array(subjects)  # (1, 7851)

    subjects_list = np.unique(subjects)
    n_subjects = subjects_list.shape[-1]  # 17
    print("Number of different subject: {}".format(n_subjects))

    # iterate over every subject and use it as a test subject
    for subject in subjects_list:
        subjects_iter = subjects
        subjects_list_iter = subjects_list

        # mask
        mask = np.isin(subjects_iter, subject)

        # determine index of the test subject and delete it from the subjects list
        idx_test = np.where(mask.reshape(-1))  # subjects_iter is 2 dim, transfer to 1 dim

        # subjects_iter = np.delete(subjects_iter, np.where(idx_test))
        # subjects_list_iter = np.delete(subjects_list_iter, np.where(mask))
        subjects_list_iter = np.setdiff1d(subjects_list_iter, subject)

        # split remaining subjects into training and validation set
        subjects_train, subjects_val = train_test_split(subjects_list_iter, test_size=0.2)

        idx_train = np.where(np.isin(subjects_iter, subjects_train))[-1]
        idx_val = np.where(np.isin(subjects_iter, subjects_val))[-1]

        # if personalization is enabled: assign some data from the test subjects to the training set
        if opt.PerformPersonalization:
            # choose data randomly or use first 20 % of the test subject's data
            if opt.RandomPick:
                idx_test, idx_add_train = train_test_split(idx_test, test_size=0.2)
                idx_train = np.concatenate((idx_train, idx_add_train), axis=0)
            else:
                N_add_train = np.round(idx_test.shape[0] * 0.2).astype(int)
                idx_add_train, idx_test = np.split(idx_test, [N_add_train])
                idx_train = np.concatenate((idx_train, idx_add_train), axis=0)

        bp_train = BP[idx_train]
        rppg_train = rppg[idx_train, :]
        idx_shuffle = np.random.permutation(bp_train.shape[0] - 1)
        bp_train = bp_train[idx_shuffle]
        rppg_train = rppg_train[idx_shuffle, :]

        bp_val = BP[idx_val]
        rppg_val = rppg[idx_val, :]
        bp_test = BP[idx_test]
        rppg_test = rppg[idx_test, :]

        # to tensor
        train_data = TDataset(rppg_train, bp_train)
        val_data = TDataset(rppg_val, bp_val)
        test_data = TDataset(rppg_test, bp_test)

        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)

        # predict test samples' bp of MIMIC_III_ppg_test before fine-tune
        model.load_state_dict(torch.load('save/resnet18_202307141720/best_w.pth')['state_dict'])  # 50
        pre_result = bp_predict(model, test_loader)

        # freeze layers
        for param in list(model.parameters())[:-opt.freeze_layers]:
            param.requires_grad = False

        lr = opt.lr
        best_loss = 1e-3
        start_epoch = 1
        stage = 1

        step = [int(opt.n_epochs * 0.3), int(opt.n_epochs * 0.6), int(opt.n_epochs * 0.9)]
        weight_decay = 2

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        states = []

        model_save_dir = 'save/finetune_result/' + str(subject)
        # os.makedirs(model, exist_ok=True)

        print("Now it's turn to subject_{}".format(str(subject)))
        for epoch in range(start_epoch, opt.n_epochs):
            since = time.time()

            train_loss = train_epoch(model, optimizer, train_loader)
            val_loss = val_epoch(model, optimizer, val_loader)

            print('#epoch: %02d stage: %d train_loss: %.3e val_loss: %0.3e time: %s\n'
                  % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)))

            writer = SummaryWriter(model_save_dir)
            writer.add_scalar('train_loss', train_loss, epoch)  # add_scalar 添加标量
            writer.add_scalar('val_loss', val_loss, epoch)  # add_scalar 添加标量
            writer.close()

            state = {"state_dict": model.state_dict(), "epoch": epoch,
                     "loss": val_loss, 'lr': lr, 'stage': stage}

            states.append(state)
            # min(states["loss"])
            # save_ckpt(state, best_loss > val_loss, model_save_dir)
            # best_lost = min(best_loss, val_loss)

            # if epoch in step:
            #     stage += 1
            #     lr /= 10
            #
            #     print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            #     utils.adjust_learning_rate(optimizer, lr)

        poster_result = bp_predict(model, test_loader)
        pd_col_names = ['SBP_hat_pre', 'DBP_hat_pre', 'SBP_true', 'DBP_true', 'SBP_hat_post', 'DBP_hat_post']
        pd.concat([pd.DataFrame(pre_result), pd.DataFrame(poster_result[:, :-2])], axis=1).to_csv(
            "finetune_result/bp_table_{}.csv".format(str(subject)),
            header=pd_col_names,
            index=False)

        # checkpoints_path = f"./save/fine-tune/" + str(subject)
        # os.makedirs(checkpoints_path, exist_ok=True)
        # torch.save(states, checkpoints_path + '/' + 'checkpoints.pth')


def main():
    print('loading data...')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=1024, help="batch size of training")
    parser.add_argument("-r", "--lr", type=int, default=1e-6, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='resnet18', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    parser.add_argument('--N_trials', type=int, default=20, help="Number subjects used for personalization")
    parser.add_argument('--freeze_layers', type=int, default=2, help="number of layers was frozen")

    parser.add_argument('--data_file', type=str, default="rPPG_data/rPPG_bp_subject.h5", help="path of rPPG data")
    parser.add_argument('--PerformPersonalization', default=False,
                        help="if assign some data from the test subjects to the training set")
    parser.add_argument('--RandomPick', default=False,
                        help="If choose data randomly or use first 20 % of the test subject's data")

    opt = parser.parse_args()

    del_file("finetune_result")
    fine_tuning(opt)


if __name__ == '__main__':
    main()
