# -*- coding: utf-8 -*-
"""
@Time ： 2022/5/23 0:06
@Auth ： luchengyue@sz.tsinghua.edu.cn
@File ：train.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
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
import utils
from resnet18_1D import resnet18_1d
from PPG2BP_Dataset import PPG2BPDataset

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

    for (ppg, bp) in train_dataloader:
        ppg = ppg.to(device)
        ppg = ppg.unsqueeze(dim=0)
        ppg = torch.transpose(ppg, 1, 0)
        bp_hat = model(ppg).cpu()
        dbp_hat, sbp_hat = bp_hat[:, 0], bp_hat[:, 1]
        optimizer.zero_grad()

        loss_dbp = F.mse_loss(dbp_hat, bp[:, 0])
        loss_sbp = F.mse_loss(sbp_hat, bp[:, 1])

        loss = loss_dbp + loss_sbp

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()

        it_count += 1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d, loss: %.3e" % (it_count, loss.item()))

    return loss_meter / it_count


def val_epoch(model, optimizer, val_dataloader):
    model.eval()
    loss_meter, it_count = 0, 0
    with torch.no_grad():
        for (ppg, bp) in val_dataloader:
            ppg = ppg.to(device)
            ppg = ppg.unsqueeze(dim=0)
            ppg = torch.transpose(ppg, 1, 0)
            bp_hat = model(ppg).cpu()
            dbp_hat, sbp_hat = bp_hat[:, 0], bp_hat[:, 1]
            optimizer.zero_grad()

            loss_dbp = F.mse_loss(dbp_hat, bp[:, 0])
            loss_sbp = F.mse_loss(sbp_hat, bp[:, 1])

            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            it_count += 1

    return loss_meter / it_count


def train():
    print('loading data...')

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=2048, help="batch size of training")
    parser.add_argument("-t", "--type", type=str, default='cnn', help="model type")
    parser.add_argument("-m", "--model", type=str, default='v1', help="model to execute")
    opt = parser.parse_args()

    "model"
    resnet_1d = resnet18_1d()
    model = resnet_1d.to(device)

    model_save_dir = f'save/{opt.type}_{time.strftime("%Y%m%d%H%M")}'
    os.makedirs(model_save_dir, exist_ok=True)

    train_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    val_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val"
    train_data = PPG2BPDataset(train_data_path)
    val_data = PPG2BPDataset(val_data_path)

    train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    lr = 1e-3
    start_epoch = 1
    stage = 1
    step = [50, 100, 180]

    states = []

    for epoch in range(start_epoch, opt.n_epochs):
        since = time.time()
        train_loss = train_epoch(model, optimizer, train_loader, 50)
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

        if epoch in step:
            stage += 1
            lr /= 10

            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

    torch.save(states, f'./resnet18_1D_states_{opt.model}.pth')


if __name__ == '__main__':
    train()

# tensorboard --logdir=cnn_202305061217 --port=6007
