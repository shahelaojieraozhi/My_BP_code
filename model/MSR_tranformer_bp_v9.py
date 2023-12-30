# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/22 12:22
@Author  : Rao Zhi
@File    : MSR_tranformer_bp_v2.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch.nn as nn
import torch


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


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


class msr_tf_bp(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10, reduction=16):
        """
        inplanes 是提供给block的通道数，planes表示block的输出通道。
        大家知道，在做残差相加的时候，我们必须保证残差的维度与真正输出的维度相等（注意这里维度是宽高以及深度）这样我们才能把它们堆到一起，所以程序中出现了降采样操作。

        """
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(msr_tf_bp, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # attention module
        self.se = SELayer(64, reduction)
        self.se_256 = SELayer(256, reduction)

        # 网络的第一层加入注意力机制
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        # 网络的卷积层的最后一层加入注意力机制
        self.ca_x = ChannelAttention(256)
        self.sa_x = SpatialAttention()

        self.ca_y = ChannelAttention(256)
        self.sa_y = SpatialAttention()

        self.ca_z = ChannelAttention(256)
        self.sa_z = SpatialAttention()

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
        self.drop = nn.Dropout(0.2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        """
        self.avgpool = nn.AdaptiveAvgPool1d(H, w)   nn.AdaptiveAvgPool1d(1) 即为 (1, 1)
        自适应池化, 对输入信号，提供自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W， 但是输入和输出特征的数目不会变化。
        """
        self.fc = nn.Linear(256 * 3, num_classes)
        self.sigmoid = nn.Sigmoid()

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
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
        x0 = self.conv1(x0)  # (1024, 64, 438)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x0 = self.ca(x0) * x0
        x0 = self.sa(x0) * x0

        x0 = self.maxpool(x0)  # (1024, 64, 219)

        # residual = x0
        # x0 = self.se(x0)
        # x0 += residual

        x = self.layer3x3_1(x0)  # (1024, 64, 219)
        x = self.layer3x3_2(x)  # (1024, 128, 110)
        x = self.layer3x3_3(x)  # (1024, 256, 55)
        x = self.ca_x(x) * x
        x = self.sa_x(x) * x
        x = self.avgpool(x)  # (1024, 256, 1)

        y = self.layer5x5_1(x0)  # (1024, 64, 215)
        y = self.layer5x5_2(y)  # (1024, 128, 105)
        y = self.layer5x5_3(y)  # (1024, 256, 50)
        y = self.ca_y(y) * y
        y = self.sa_y(y) * y
        y = self.avgpool(y)  # (1024, 256, 1)

        z = self.layer7x7_1(x0)  # (1024, 64, 211)
        z = self.layer7x7_2(z)  # (1024, 128, 100)
        z = self.layer7x7_3(z)  # (1024, 256, 44)
        z = self.ca_z(z) * z
        z = self.sa_z(z) * z
        z = self.avgpool(z)  # (1024, 256, 1)

        out = torch.cat([x, y, z], dim=2)  # (1024, 256, 3) or # (1024, 512, 3)

        # residual_ = out
        # out = self.se_256(out)
        # out += residual_

        out = torch.transpose(out, 0, 2)  # (3, 256, 1024)
        out = torch.transpose(out, 1, 2)  # (3, 1024, 256)

        out = self.transformerEncoder(out)
        out = torch.transpose(out, 0, 1)  # (1024, 3, 256)
        out = out.reshape(out.size(0), -1)  # (batch_size, -1)

        out = self.drop(out)
        out = self.fc(out)  # (1024, 6)

        return out


if __name__ == '__main__':
    msresnet = msr_tf_bp(input_channel=3, layers=[1, 1, 1, 1], num_classes=17)
    inputs = torch.rand(1024, 3, 875)
    outputs = msresnet(inputs)
    print(outputs.size())
