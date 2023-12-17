# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/15 17:24
@Author  : Rao Zhi
@File    : train.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
from torch.utils.data import Dataset
import datetime
import os
import random
import shutil
import time
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
from torch import optim
import utils
import sys
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model.Resnet import resnet18, resnet34, resnet50, resnet101, resnet152

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class PPGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, mode="train"):
        super(PPGDataset, self).__init__()

        self.ppg = torch.load("ppg.h5")
        self.label = torch.load("label.h5")

        if mode == 'train':
            self.ppg = self.ppg[:1400]
            self.label = self.label[:1400]

        else:
            self.ppg = self.ppg[1400:]
            self.label = self.label[1400:]

    def __getitem__(self, index):
        ppg = self.ppg[index, :]
        ppg = ppg.unsqueeze(dim=0)
        target = self.label[index]

        return ppg, target

    def __len__(self):
        # return len(self.h5_path_list) * 1000
        return len(self.ppg)


# if __name__ == '__main__':
#     data = PPGDataset(mode='train')
#     datasize = len(data)
#     print(datasize)
#
#     for sample in data:
#         pulse, label = sample
#         plt.plot(pulse.squeeze())
#         plt.show()
#     # b = data[0]


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, train_dataloader, criterion, opt):
    model.train()

    acc_meter, loss_meter, it_count = 0, 0, 0
    for (ppg, label) in train_dataloader:
        ppg = ppg.to(device)
        optimizer.zero_grad()
        output = torch.sigmoid(model(ppg).cpu().squeeze())

        # loss = criterion(torch.tensor(output, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
        # loss = criterion(torch.tensor(model(ppg).cpu().squeeze(), dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
        # loss = focal_loss(torch.tensor(output, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))
        loss = criterion(output, label.float())

        # loss = torch.tensor(loss, requires_grad=True)  # 生成变量

        predict = (output >= 0.5).float()  # 二分类阈值为0.5

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()

        acc = torch.eq(predict, label).sum().float().item() / len(ppg)
        acc_meter += acc

        it_count += 1
        if it_count != 0 and it_count % opt.show_interval == 0:
            print("%d, loss: %.3e acc: %.3f" % (it_count, loss.item(), acc))

    return loss_meter / it_count, acc_meter / it_count


def val_epoch(model, val_dataloader, criterion):
    model.eval()
    acc_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for (ppg, label) in val_dataloader:
            ppg = ppg.to(device)
            output = torch.sigmoid(model(ppg).cpu().squeeze())

            loss = criterion(output, label.float())

            predict = (output >= 0.5).float()

            loss_meter += loss.item()
            # predict = output.argmax(dim=1)
            acc = torch.eq(predict, label).sum().float().item() / len(ppg)
            acc_meter += acc
            it_count += 1

    return loss_meter / it_count, acc_meter / it_count


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
    # model = resnet50(input_c=1 if input_channel == 1 else 3, num_classes=2)
    model = resnet18(input_c=1 if input_channel == 1 else 3, num_classes=1)

    model = model.to(device)
    model_save_dir = f'save/{opt.model}_{opt.describe}_{time.strftime("%Y%m%d%H")}'
    os.makedirs(model_save_dir, exist_ok=True)

    """load data"""
    print('loading data...')
    train_data = PPGDataset('train')
    val_data = PPGDataset('val')

    train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    states = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, opt.n_epochs):
        since = time.time()

        train_loss, train_acc = train_epoch(model, optimizer, train_loader, criterion, opt)
        val_loss, val_acc = val_epoch(model, val_loader, criterion)

        print('#epoch: %02d stage: %d train_loss: %.3e train_acc: %.3f  val_loss: %0.3e val_acc: %.3f time: %s'
              % (epoch, stage, train_loss, train_acc, val_loss, val_acc, utils.print_time_cost(since)), end='\n')

        # Determine approximate time left
        epoch_done = opt.n_epochs - epoch

        # Print log
        sys.stdout.write("\rETA(left time): %s" % utils.left_time(since, epoch_done))

        writer = SummaryWriter(model_save_dir)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
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
    parser.add_argument("-t", "--model", type=str, default='abnormal_pulse_detection', help="model type")
    parser.add_argument("-d", "--describe", type=str, default='total=1798', help="describe for this model")
    parser.add_argument("-n", "--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=128, help="batch size of training")
    parser.add_argument("-bl", "--best_loss", type=int, default=1e3, help="best_loss")
    parser.add_argument("-lr", "--lr", type=int, default=1e-4, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=0, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument("-ds", "--decay_step", type=list, default=[100], help="decay step list of learning rate")
    parser.add_argument("-wd", "--weight_decay", type=int, default=1e-3, help="weight_decay")
    parser.add_argument('--using_derivative', default=False, help='using derivative of PPG or not')
    parser.add_argument('--show_interval', type=int, default=3, help='how long to show the loss value')
    parser.add_argument('--loss_func', type=str, default='HuberLoss',
                        choices=('SmoothL1Loss', 'mse', 'bp_bucketing_loss', 'HuberLoss'),
                        help='which loss function is selected')
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="para of HuberLoss and SmoothL1Loss")
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    train(args)

# tensorboard --logdir=cnn_202305061217 --port=6007
# tensorboard --logdir=save
