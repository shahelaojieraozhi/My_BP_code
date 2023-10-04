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
from model.resnet18_1D import resnet18_1d
from PPG2BP_Dataset_finetune import TDataset

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)


def del_file(path_data):
    """delete all directory in the current path"""
    for element in os.listdir(path_data):
        file_data = path_data + "\\" + element
        if os.path.isfile(file_data):
            os.remove(file_data)
        else:
            del_file(file_data)


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
        # ppg = ppg.squeeze(1)

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


def personalize_train(opt):
    # model
    resnet_1d = resnet18_1d()
    model = resnet_1d.to(device)

    # model_save_dir = f'save/{opt.type}_{time.strftime("%Y%m%d%H%M")}'
    # os.makedirs(model_save_dir, exist_ok=True)

    # load the test dataset of MIMIC-iii (here, it is regarded as the train dataset)
    # data = np.load(r'G:\Blood_Pressure_dataset\cvprw\08-11-2023_pers_dataset.npz')
    data = np.load('reshaped_test_dataset.npz')

    ppg = data['arr_0']  # shape = (250000, 875)
    bp = data['arr_1']  # shape = (250000, 2)
    subject_idx = data['arr_2']  # shape = (250000)  # shape = (250000)
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
        train_data = TDataset(ppg_train, bp_train)
        val_data = TDataset(ppg_val, bp_val)
        test_data = TDataset(ppg_test, bp_test)

        train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=0)

        # predict test samples' bp of MIMIC_III_ppg_test before fine-tune
        model.load_state_dict(torch.load('save/resnet18_202307141720/best_w.pth')['state_dict'])  # 50
        pre_result = bp_predict(model, test_loader)

        # for name, param in model.named_parameters():
        #     # print(name)
        #     # print(param)
        #     print(name, "   ", param.shape)

        # a1 = list(model.parameters())
        # a = list(model.parameters())[:-8]

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

        model_save_dir = 'save/fine-tune/' + str(subject) + '/'
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
            save_ckpt(state, best_loss > val_loss, model_save_dir)
            # best_lost = min(best_loss, val_loss)

            if epoch in step:
                stage += 1
                lr /= 10

                print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
                utils.adjust_learning_rate(optimizer, lr)

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
    parser.add_argument("-n", "--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size of training")
    parser.add_argument("-r", "--lr", type=int, default=1e-6, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='resnet18', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    parser.add_argument('--N_trials', type=int, default=20, help="Number subjects used for personalization")
    parser.add_argument('--freeze_layers', type=int, default=8, help="number of layers was frozen")
    opt = parser.parse_args()

    del_file("finetune_result")
    personalize_train(opt)


if __name__ == '__main__':
    main()

# tensorboard --logdir=cnn_202305061217 --port=6007
