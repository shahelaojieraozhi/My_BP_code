# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/4 20:24
@Author  : Rao Zhi
@File    : ppg2bp_net.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels * BasicBlock.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels * BasicBlock.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv1d(
            1, 64, kernel_size=13, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):  # (2048, 1, 875)
        x = self.conv1(x)  # (2048, 64, 438)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (2048, 64, 219)

        x = self.layer1(x)  # (2048, 64, 219)
        x = self.layer2(x)  # (2048, 128, 110)
        x = self.layer3(x)  # (2048, 256, 55)
        x = self.layer4(x)  # (2048, 512, 28)

        x = self.avgpool(x)  # (2048, 512, 1)
        x = torch.flatten(x, 1)  # (2048, 512)
        x = self.dropout(x)
        x = self.fc(x)  # (2048, 2)
        # x = self.sigmoid(x)       # (2048, 2)

        return x


def resnet18_1d(num_classes=2):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34_1d(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


if __name__ == '__main__':
    # Create an instance of the ResNet18 model
    # model = resnet18_1d()
    model = resnet34_1d()

    # Generate a random input signal of length 875
    input_signal = torch.randn(100, 1, 875)

    # Pass the input through the model
    output = model(input_signal)

    print(output.shape)  # Output shape: [1, 2] representing two values
