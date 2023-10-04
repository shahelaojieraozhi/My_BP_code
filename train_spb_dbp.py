# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/12 9:00
@Author  : Rao Zhi
@File    : train_spb_dbp.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

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
from model.resnet18_1D import resnet18_1d, resnet34_1d
from model.Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from PPG2BP_Dataset_v2 import PPG2BPDataset
# from Transformer_reg import TF
from Transformer_reg_v2 import RegressionTransformer
from model.bp_MSR_Net import MSResNet

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


def train(opt):
    # load param
    best_lost = opt.best_lost
    lr = opt.lr
    start_epoch = opt.start_epoch
    stage = opt.stage
    step = opt.decay_step
    weight_decay = opt.weight_decay

    "load model"
    # model = TF(in_features=875, drop=0.).to(device)
    # model = RegressionTransformer(input_dim=875, output_dim=2)
    # model = resnet34_1d().to(device)

    # resnet_1d = resnet18_1d()
    # resnet_1d = resnet18()
    resnet_1d = resnet50()
    model = resnet_1d.to(device)

    # bp_msr_net = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=2)
    # model = bp_msr_net.to(device)

    # model_save_dir = f'save/{opt.type}_{time.strftime("%Y%m%d%H%M")}'
    model_save_dir = f'save/{opt.model}_{opt.describe}_{time.strftime("%Y%m%d%H")}'
    os.makedirs(model_save_dir, exist_ok=True)

    """load data"""
    print('loading data...')
    train_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\train"
    val_data_path = "G:\\Blood_Pressure_dataset\\cvprw\\h5_record\\val"
    train_data = PPG2BPDataset(train_data_path)
    val_data = PPG2BPDataset(val_data_path)

    train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        save_ckpt(state, best_lost > val_loss, model_save_dir)
        best_lost = min(best_lost, val_loss)

        if epoch in step:
            stage += 1
            lr /= 10

            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

    torch.save(states, f'./save/resnet18_1D_states.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model", type=str, default='resnet50', help="model type")
    parser.add_argument("-n", "--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=2048, help="batch size of training")
    parser.add_argument("-d", "--describe", type=str, default='lr=1e-3', help="describe for this model")
    parser.add_argument("-bl", "--best_lost", type=int, default=1e3, help="best_lost")
    parser.add_argument("-lr", "--lr", type=int, default=1e-3, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=1, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument("-ds", "--decay_step", type=list, default=[5, 15, 25], help="decay step list of learning rate")
    parser.add_argument("-wd", "--weight_decay", type=int, default=2, help="weight_decay")
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    train(args)

# tensorboard --logdir=cnn_202305061217 --port=6007
# tensorboard --logdir=add_normal_res_18 --port=6007
