# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/7/12 9:00
@Author  : Rao Zhi
@File    : train.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import datetime
import os
import random
import shutil
import time
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from model.Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from PPG2BP_Dataset import PPG2BPDataset, use_derivative
from model.bp_MSR_Net import MSResNet

# from model.bpnet_cvprw import resnet50
# from model.resnet1d import resnet50
# from model.MSR_tranformer_bp import msr_tf_bp
# from model.MSR_tranformer_bp_ppg_segment import msr_tf_bp_ppg_segment
# from model.resnet18_1D_segment import resnet18_1d

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_calculate(pre, label, loss_name='mse'):
    if loss_name == 'mse':
        return F.mse_loss(pre, label)
    elif loss_name == "SmoothL1Loss":
        smooth_l1_loss = torch.nn.SmoothL1Loss()
        return smooth_l1_loss(pre, label)


def seed_torch(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To prohibit hash randomization and make the experiment replicable
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, train_dataloader, opt):
    model.train()

    loss_meter, loss_sbp_meter, loss_dbp_meter, it_count = 0, 0, 0, 0
    for (ppg, sbp, dbp) in train_dataloader:
        ppg = ppg.to(device)
        ppg = use_derivative(ppg) if opt.using_derivative else ppg
        bp_hat = model(ppg).cpu()
        sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
        optimizer.zero_grad()

        loss_sbp = loss_calculate(sbp_hat, sbp, opt.loss_func)
        loss_dbp = loss_calculate(dbp_hat, dbp, opt.loss_func)
        # loss_sbp = loss_function(sbp_hat, bp[:, 0])
        # loss_dbp = loss_function(dbp_hat, bp[:, 1])

        loss = loss_dbp + loss_sbp
        loss_sbp_meter += loss_sbp.item()
        loss_dbp_meter += loss_dbp.item()

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()

        it_count += 1
        if it_count != 0 and it_count % opt.show_interval == 0:
            print("%d, loss: %.3e" % (it_count, loss.item()))
            # print("%d, whole loss: %.3e, sbp loss: %.3e, dbp loss: %.3e" % (
            #     it_count, loss.item(), loss_sbp.item(), loss_dbp.item()))

    return loss_meter / it_count, loss_sbp_meter / it_count, loss_dbp_meter / it_count


def val_epoch(model, val_dataloader, opt):
    model.eval()
    loss_meter, loss_sbp_meter, loss_dbp_meter, it_count = 0, 0, 0, 0
    with torch.no_grad():
        for (ppg, sbp, dbp) in val_dataloader:
            ppg = ppg.to(device)
            # ppg = ppg.unsqueeze(dim=0)
            # ppg = torch.transpose(ppg, 1, 0)
            ppg = use_derivative(ppg) if opt.using_derivative else ppg
            bp_hat = model(ppg).cpu()
            sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]

            loss_sbp = loss_calculate(sbp_hat, sbp, opt.loss_func)
            loss_dbp = loss_calculate(dbp_hat, dbp, opt.loss_func)

            loss = loss_dbp + loss_sbp
            loss_meter += loss.item()
            loss_sbp_meter += loss_sbp.item()
            loss_dbp_meter += loss_dbp.item()

            it_count += 1

    return loss_meter / it_count, loss_sbp_meter / it_count, loss_dbp_meter / it_count


def train(opt):
    "load param"
    best_loss = opt.best_loss
    lr = opt.lr
    start_epoch = opt.start_epoch
    stage = opt.stage
    step = opt.decay_step
    weight_decay = opt.weight_decay
    input_channel = 3 if opt.using_derivative else 1

    "load model"
    # model = resnet50(num_input_channels=1, num_classes=2)
    # model = MSResNet(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)
    # model = resnet18()
    # model = resnet50(input_c=3, num_classes=2)
    model = resnet50(input_c=1 if input_channel == 1 else 3, num_classes=2)
    model = model.to(device)

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
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    states = []

    for epoch in range(start_epoch, opt.n_epochs):
        since = time.time()

        train_loss, train_sbp_loss, train_dbp_loss, = train_epoch(model, optimizer, train_loader, opt)
        val_loss, val_sbp_loss, val_dbp_loss = val_epoch(model, val_loader, opt)

        print('#epoch: %02d stage: %d train_loss: %.3e val_loss: %0.3e time: %s'
              % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)), end='\n')
        print('#train_sbp_loss: %.3e train_dbp_loss: %0.3e val_sbp_loss: %.3e val_dbp_loss: %.3e\n'
              % (train_sbp_loss, train_dbp_loss, val_sbp_loss, val_dbp_loss))

        writer = SummaryWriter(model_save_dir)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_sbp_loss', train_sbp_loss, epoch)
        writer.add_scalar('val_sbp_loss', val_sbp_loss, epoch)
        writer.add_scalar('train_dbp_loss', train_dbp_loss, epoch)
        writer.add_scalar('val_dbp_loss', val_dbp_loss, epoch)
        writer.close()

        state = {"state_dict": model.state_dict(), "epoch": epoch,
                 "loss": val_loss, 'lr': lr, 'stage': stage}
        states.append(state)

        save_ckpt(state, best_loss > val_loss, model_save_dir)
        best_loss = min(best_loss, val_loss)

        if epoch in step:
            stage += 1
            lr /= 10

            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)

    # torch.save(states, f'./save/resnet18_1D_states.pth')
    print(datetime.datetime.now())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--model", type=str, default='res50', help="model type")
    parser.add_argument("-d", "--describe", type=str, default='reproduce', help="describe for this model")
    parser.add_argument("-n", "--n_epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=256, help="batch size of training")
    parser.add_argument("-bl", "--best_loss", type=int, default=1e3, help="best_loss")
    parser.add_argument("-lr", "--lr", type=int, default=1e-3, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=1, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument("-ds", "--decay_step", type=list, default=[100], help="decay step list of learning rate")
    parser.add_argument("-wd", "--weight_decay", type=int, default=2, help="weight_decay")
    parser.add_argument('--using_derivative', default=False, help='using derivative of PPG or not')
    parser.add_argument('--show_interval', type=int, default=50, help='how long to show the loss value')
    parser.add_argument('--loss_func', type=str, default='SmoothL1Loss', help='which loss function is selected')
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    train(args)

# tensorboard --logdir=cnn_202305061217 --port=6007
