# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/15 9:05
@Author  : Rao Zhi
@File    : FocalLoss_reg.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        # input: predictions, target: ground truth
        ce_loss = F.smooth_l1_loss(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()

        if self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss
        else:
            raise ValueError("Invalid reduction option. Use 'mean' or 'sum'.")


