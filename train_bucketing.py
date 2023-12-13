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
import torch.nn as nn
from model.Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from PPG2BP_Dataset_bucketing import PPG2BPDataset, use_derivative

# from model.bp_MSR_Net import MSResNet

# from model.bpnet_cvprw import resnet50
# from model.resnet1d import resnet50
from model.MSR_tranformer_bp_v2 import msr_tf_bp

# from model.MSR_tranformer_bp_ppg_segment import msr_tf_bp_ppg_segment
# from model.resnet18_1D_segment import resnet18_1d

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


def multi_task_loss_v3(reg_prediction, reg_target, class_prediction, class_target, which_bp):

    """测试了，无效！！！"""
    # reg_loss_func = torch.nn.SmoothL1Loss()
    # smooth_l1_loss = reg_loss_func(reg_prediction, reg_target)

    if which_bp == 'sbp':
        # sbp
        value = torch.Tensor([75, 85, 95, 105, 115, 125, 135, 145, 155, 165])
        pro = F.softmax(class_prediction)
        pre_final = torch.matmul(pro, value)
        bucket_loss = F.mse_loss(pre_final, reg_target)
    else:
        # dbp
        value = torch.Tensor([40, 50, 60, 70, 80])
        pro = F.softmax(class_prediction)
        pre_final = torch.matmul(pro, value)
        bucket_loss = F.mse_loss(pre_final, reg_target)

    return bucket_loss


def multi_task_loss(reg_prediction, reg_target, class_prediction, class_target, which_bp):
    # class_targets, reg_targets = targets[:, 0].long(), targets[:, 1:].squeeze()

    classification_loss = F.cross_entropy(class_prediction, class_target)

    # predicted_classes = torch.argmax(predictions[:, :len(boundaries)], dim=1)

    # a = predictions[:, -1]
    # b = reg_targets

    # if torch.unique(class_target).numel() > 6:
    if which_bp == 'sbp':
        # sbp
        regression_loss = F.mse_loss(reg_prediction, reg_target) / (len(reg_target) * (165 - 75))
    else:
        # dbp
        regression_loss = F.mse_loss(reg_prediction, reg_target) / (len(reg_target) * (80 - 40))

    total_loss = classification_loss + regression_loss

    return total_loss


def multi_task_loss_v2(reg_prediction, reg_target, class_prediction, class_target, which_bp):
    reg_loss_func = torch.nn.SmoothL1Loss()
    smooth_l1_loss = reg_loss_func(reg_prediction, reg_target)

    if which_bp == 'sbp':
        # sbp
        value = torch.Tensor([75, 85, 95, 105, 115, 125, 135, 145, 155, 165])
        pro = F.softmax(class_prediction)
        pre_final = torch.matmul(pro, value)
        bucket_loss = F.mse_loss(pre_final, reg_target)
    else:
        # dbp
        value = torch.Tensor([40, 50, 60, 70, 80])
        pro = F.softmax(class_prediction)
        pre_final = torch.matmul(pro, value)
        bucket_loss = F.mse_loss(pre_final, reg_target)

    return smooth_l1_loss + bucket_loss


def loss_calculate(pre, label, class_label_hat, class_label, loss_name='mse', which='sbp'):
    if loss_name == 'mae':
        mae_loss = nn.L1Loss()
        return mae_loss(pre, label)

    elif loss_name == 'mse':
        mse_loss = nn.MSELoss()
        return mse_loss(pre, label)

    elif loss_name == 'SmoothL1Loss':
        smooth_l1_loss = torch.nn.SmoothL1Loss()
        return smooth_l1_loss(pre, label)

    elif loss_name == 'multi_task_loss':
        # return multi_task_loss(pre, label, class_label_hat, class_label, which)
        return multi_task_loss_v2(pre, label, class_label_hat, class_label, which)

    # elif loss_name == 'bp_bucketing_loss':
    #     bp_bucketing_loss = BucketingLoss(pre, label, class_label, class_label_hat)
    #     return bp_bucketing_loss


def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, train_dataloader, opt):
    model.train()

    mae_loss, loss_meter, loss_sbp_meter, loss_dbp_meter, it_count = 0, 0, 0, 0, 0
    for (ppg, sbp, dbp, sbp_class, dbp_class) in train_dataloader:
        ppg = ppg.to(device)
        ppg = use_derivative(ppg) if opt.using_derivative else ppg
        bp_hat = model(ppg).cpu()
        sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
        sbp_class_hat, dbp_class_hat = bp_hat[:, 2:12], bp_hat[:, 12:]
        optimizer.zero_grad()

        mae_loss_sbp = F.l1_loss(sbp_hat, sbp)
        mae_loss_dbp = F.l1_loss(dbp_hat, dbp)

        loss_sbp = loss_calculate(sbp_hat, sbp, sbp_class_hat, sbp_class, opt.loss_func, 'sbp')
        loss_dbp = loss_calculate(dbp_hat, dbp, dbp_class_hat, dbp_class, opt.loss_func, 'dbp')
        # loss_sbp = loss_function(sbp_hat, bp[:, 0])
        # loss_dbp = loss_function(dbp_hat, bp[:, 1])

        loss = loss_dbp + loss_sbp
        loss_sbp_meter += loss_sbp.item()
        loss_dbp_meter += loss_dbp.item()

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        mae_loss += (mae_loss_sbp + mae_loss_dbp).item()

        it_count += 1
        if it_count != 0 and it_count % opt.show_interval == 0:
            print("%d, loss: %.3e, MAE loss: %.3e" % (it_count, loss.item(), (mae_loss_sbp + mae_loss_dbp).item()))
            # print("%d, whole loss: %.3e, sbp loss: %.3e, dbp loss: %.3e" % (
            #     it_count, loss.item(), loss_sbp.item(), loss_dbp.item()))

    return loss_meter / it_count, loss_sbp_meter / it_count, loss_dbp_meter / it_count, mae_loss / it_count


def val_epoch(model, val_dataloader, opt):
    model.eval()
    mae_loss, loss_meter, loss_sbp_meter, loss_dbp_meter, it_count = 0, 0, 0, 0, 0
    for (ppg, sbp, dbp, sbp_class, dbp_class) in val_dataloader:
        ppg = ppg.to(device)
        ppg = use_derivative(ppg) if opt.using_derivative else ppg
        bp_hat = model(ppg).cpu()
        sbp_hat, dbp_hat = bp_hat[:, 0], bp_hat[:, 1]
        sbp_class_hat, dbp_class_hat = bp_hat[:, 2:12], bp_hat[:, 12:]

        mae_loss_sbp = F.l1_loss(sbp_hat, sbp)
        mae_loss_dbp = F.l1_loss(dbp_hat, dbp)

        loss_sbp = loss_calculate(sbp_hat, sbp, sbp_class_hat, sbp_class, opt.loss_func, 'sbp')
        loss_dbp = loss_calculate(dbp_hat, dbp, dbp_class_hat, dbp_class, opt.loss_func, 'dbp')

        loss = loss_dbp + loss_sbp
        loss_meter += loss.item()
        loss_sbp_meter += loss_sbp.item()
        loss_dbp_meter += loss_dbp.item()
        mae_loss += (mae_loss_sbp + mae_loss_dbp).item()

        it_count += 1

    return loss_meter / it_count, loss_sbp_meter / it_count, loss_dbp_meter / it_count, mae_loss / it_count


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

    # model = resnet50(input_c=1 if input_channel == 1 else 3, num_classes=2)
    # model = resnet18(input_c=1 if input_channel == 1 else 3, num_classes=2)
    model = msr_tf_bp(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=17)
    model = model.to(device)

    # model_save_dir = f'save/{opt.type}_{time.strftime("%Y%m%d%H%M")}'
    model_save_dir = f'save/{opt.model}_{opt.describe}_{time.strftime("%Y%m%d%H")}'
    os.makedirs(model_save_dir, exist_ok=True)

    """load data"""
    print('loading data...')
    train_data = PPG2BPDataset('train')
    val_data = PPG2BPDataset('val')

    train_loader = DataLoader(train_data, batch_size=opt.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=opt.batch, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    states = []

    for epoch in range(start_epoch, opt.n_epochs):
        since = time.time()

        train_loss, train_sbp_loss, train_dbp_loss, train_mae_loss = train_epoch(model, optimizer, train_loader, opt)
        val_loss, val_sbp_loss, val_dbp_loss, val_mae_loss = val_epoch(model, val_loader, opt)

        print('#epoch: %02d stage: %d train_loss: %.3e val_loss: %0.3e time: %s'
              % (epoch, stage, train_loss, val_loss, utils.print_time_cost(since)), end='\n')
        print(
            '#train_sbp_loss: %.3e train_dbp_loss: %0.3e val_sbp_loss: %.3e val_dbp_loss: %.3e train_mae_loss: %.3e val_mae_loss: %.3e\n'
            % (train_sbp_loss, train_dbp_loss, val_sbp_loss, val_dbp_loss, train_mae_loss, val_mae_loss))

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
    parser.add_argument("-t", "--model", type=str, default='msr_tf_bp', help="model type")
    parser.add_argument("-d", "--describe", type=str, default='multi_task_v1', help="describe for this model")
    parser.add_argument("-n", "--n_epochs", type=int, default=60, help="number of epochs of training")
    parser.add_argument("-b", "--batch", type=int, default=1024, help="batch size of training")
    parser.add_argument("-bl", "--best_loss", type=int, default=1e3, help="best_loss")
    parser.add_argument("-lr", "--lr", type=int, default=1e-3, help="learning rate")
    parser.add_argument("-se", "--start_epoch", type=int, default=1, help="start_epoch")
    parser.add_argument("-st", "--stage", type=int, default=1, help="stage")
    parser.add_argument("-ds", "--decay_step", type=list, default=[100], help="decay step list of learning rate")
    parser.add_argument("-wd", "--weight_decay", type=int, default=1e-3, help="weight_decay")
    parser.add_argument('--using_derivative', default=True, help='using derivative of PPG or not')
    parser.add_argument('--show_interval', type=int, default=50, help='how long to show the loss value')
    parser.add_argument('--loss_func', type=str, default='multi_task_loss',
                        choices=('SmoothL1Loss', 'mse', 'bp_bucketing_loss', 'multi_task_loss'),
                        help='which loss function is selected')
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    train(args)

# tensorboard --logdir=cnn_202305061217 --port=6007
