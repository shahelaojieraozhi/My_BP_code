# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code
@Time    : 2023/10/7 8:28
@Author  : Rao Zhi
@File    : random_seeds_set.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm
@ref: javis
"""
import random
import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score
from torch import nn


# 计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


# 打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


def left_time(since, done_epoch):
    time_elapsed = (time.time() - since) * done_epoch
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()


def seed_torch(seed=1029):
    """
    参考:https://blog.csdn.net/john_bh/article/details/107731443
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # To disable hash randomization and make the experiment reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
