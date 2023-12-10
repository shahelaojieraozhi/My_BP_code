# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/11/24 16:20
@Author  : Rao Zhi
@File    : cnn-lstm.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch.nn as nn
import torch


# Define Neural Network Architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Sequential block of layer1
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3))

        self.adaptive = nn.AdaptiveMaxPool1d(4)

        # lstm and fully connected layer
        # self.lstm = nn.LSTM(256, 56)
        self.lstm = nn.LSTM(1024, 56)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # x = x.unsqueeze(-2)
        out = self.layer1(x)
        out = self.layer2(out)

        # adaptive maxpool
        out = self.adaptive(out)
        # print(out.shape)

        # flatten
        out = out.reshape(out.size(0), -1)
        # print(out1.shape)

        # lstm layer
        out = out.unsqueeze(0)
        out, hid = self.lstm(out)
        # print(out.shape)

        out = out.squeeze(0)

        # output layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.squeeze(-1)
        print(out.shape)
        return out


if __name__ == '__main__':
    model = ConvNet()
    inputs = torch.rand(256, 1, 128)
    outputs = model(inputs)
    print(outputs.size())


