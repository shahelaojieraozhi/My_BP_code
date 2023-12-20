# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/19 14:50
@Author  : Rao Zhi
@File    : 1D_resnet_attention.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        avg_pool = F.adaptive_avg_pool1d(x, 1)

        # 第一个卷积层
        out = self.conv1(x)
        out = F.relu(out)

        # 第二个卷积层
        out = self.conv2(out)

        # 加权
        out = out + avg_pool

        # 通过Sigmoid激活函数
        out = self.sigmoid(out)

        # 与原始输入相乘
        out = x * out

        return out


class ResNetWithAttention(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNetWithAttention, self).__init__()

        # 1D卷积层
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 注意力模块
        self.attention = AttentionBlock(64)

        # ResNet的基本块
        self.resnet_blocks = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64)
        )

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 注意力模块
        x = self.attention(x)

        # ResNet的基本块
        x = self.resnet_blocks(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x


if __name__ == '__main__':
    # 创建模型
    in_channels = 1  # 输入通道数
    num_classes = 2  # 输出类别数
    model = ResNetWithAttention(in_channels, num_classes)

    # msresnet = msr_tf_bp(input_channel=3, layers=[1, 1, 1, 1], num_classes=2)
    inputs = torch.rand(1024, 1, 875)
    outputs = model(inputs)
    print(outputs.size())




