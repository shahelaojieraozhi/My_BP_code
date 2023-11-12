# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/22 12:22
@Author  : Rao Zhi
@File    : MSR_tranformer_bp.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import math
import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)  # inplanes3=64, planes=64, stride=2
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # (1024, 64, 128)

        out = self.conv1(x)  # (1024, 64, 64)
        out = self.bn1(out)  # (1024, 64, 64)
        out = self.relu(out)

        out = self.conv2(out)  # (1024, 64, 64)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # (1024, 64, 64)

        out += residual  # (1024, 64, 64)
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)
        # out += residual

        return out1


class msr_tf_bp_ppg_segment(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        """
        inplanes 是提供给block的通道数，planes表示block的输出通道。
        大家知道，在做残差相加的时候，我们必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）这样我们才能把它们堆到一起，所以程序中出现了降采样操作。

        """
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(msr_tf_bp_ppg_segment, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=1)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        # self.maxpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=1)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        # self.maxpool5 = nn.AvgPool1d(kernel_size=11, stride=1, padding=0)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=1)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        # self.maxpool7 = nn.AvgPool1d(kernel_size=6, stride=1, padding=0)

        # self.drop = nn.Dropout(p=0.2)
        self.drop = nn.Dropout(0.5)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        """
        self.avgpool = nn.AdaptiveAvgPool1d(H, w)   nn.AdaptiveAvgPool1d(1) 即为 (1, 1)
        自适应池化, 对输入信号，提供自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W， 但是输入和输出特征的数目不会变化。
        """
        self.fc1 = nn.Linear(128*7*3, 128*7)
        self.fc2 = nn.Linear(128*7, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):  # (1024, 1, 512)
        batch_size = x0.size(0)  # batch_size = b
        channel_size = x0.size(1)
        time_steps = 7

        # k = w.view(batch_size * time_steps, w.size(1), int(w.size(2) / time_steps))    # one step

        x0 = x0.view(batch_size, channel_size, int(x0.size(2) / time_steps), time_steps)
        # segment ppg   shape:(b, c, 125, 7)
        x0 = x0.transpose(1, 3)  # (b, 7, 125, c)
        x0 = x0.transpose(2, 3)  # (b, 7, c, 125)
        x0 = x0.contiguous().view(batch_size * time_steps, x0.size(2), x0.size(3))
        # w ori:(b, 7, c, 125)   (b * 7, c, 125)  (35, 3, 125)

        x0 = self.conv1(x0)  # (1024, 64, 256)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)  # (1024, 64, 128)

        x = self.layer3x3_1(x0)  # (1024, 64, 64)
        x = self.layer3x3_2(x)  # (1024, 128, 32)
        # x = self.layer3x3_3(x)  # (1024, 256, 16)
        # x = self.layer3x3_4(x)
        # x = self.maxpool3(x)  # (1024, 256, 1)
        x = self.avgpool(x)  # (1024, 256, 1)

        y = self.layer5x5_1(x0)  # (1024, 64, 61)
        y = self.layer5x5_2(y)  # (1024, 128, 28)
        # y = self.layer5x5_3(y)  # (1024, 256, 11)
        # y = self.layer5x5_4(y)
        # y = self.maxpool5(y)  # (1024, 256, 1)
        y = self.avgpool(y)  # (1024, 256, 1)

        z = self.layer7x7_1(x0)  # (1024, 64, 58)
        z = self.layer7x7_2(z)  # (1024, 128, 23)
        # z = self.layer7x7_3(z)  # (1024, 256, 6)
        # z = self.layer7x7_4(z)
        # z = self.maxpool7(z)  # (1024, 256, 1)
        z = self.avgpool(z)  # (1024, 256, 1)

        out = torch.cat([x, y, z], dim=1)  # (b, 128, 3)
        out = out.view(batch_size, time_steps, -1)  # (b, 7, 64)
        out = out.view(batch_size, -1)  # (b, 7, 64)
        out = self.drop(out)
        out = self.fc1(out)  # (1024, 6)
        out = self.drop(out)
        out = self.fc2(out)  # (1024, 6)
        out = self.drop(out)
        out = self.fc3(out)  # (1024, 6)

        return out


if __name__ == '__main__':
    msresnet = msr_tf_bp_ppg_segment(input_channel=3, layers=[1, 1, 1, 1], num_classes=2)
    inputs = torch.rand(5, 3, 875)
    outputs = msresnet(inputs)
    print(outputs.size())