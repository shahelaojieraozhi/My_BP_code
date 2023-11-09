# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/22 17:30
@Author  : Rao Zhi
@File    : bp_former.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch.nn as nn
import math
import torch

__all__ = ['resnet_lstm_mitbih']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_1D(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

        out += residual
        out = self.relu(out)

        return out


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3x3_1D(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1D(planes, planes)
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

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=2, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for ecg model
        if block_name.lower() == 'basicblock':
            block = BasicBlock
            block_ecg = BasicBlock1D
        elif block_name.lower() == 'bottleneck':
            block = Bottleneck
            block_ecg = Bottleneck1D
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.batch_size = None
        self.block = block

        self.inplanes = 16
        self.inplanes_ecg = 16
        self.fc = nn.Linear(64 * block.expansion, 32 * block.expansion)
        self.fc_final = nn.Linear(32 * block.expansion, num_classes)
        self.time_steps = 7
        self.hidden_cell = None
        self.lstm_ecg = nn.LSTM(input_size=64 * block.expansion, hidden_size=64 * block.expansion, num_layers=2,
                                batch_first=True)

        self.conv1ecg = nn.Conv1d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1ecg = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1ecg = self._make_layer_1d(block_ecg, 16, 3)
        self.layer2ecg = self._make_layer_1d(block_ecg, 32, 3, stride=2)
        self.layer3ecg = self._make_layer_1d(block_ecg, 64, 3, stride=2)
        self.avgpoolecg = nn.AdaptiveAvgPool1d(output_size=(1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=64 * block.expansion, nhead=8)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.drop = nn.Dropout(0.5)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer_1d(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_ecg != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes_ecg, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes_ecg, planes, stride, downsample))
        self.inplanes_ecg = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_ecg, planes))

        return nn.Sequential(*layers)

    def forward(self, w):  # (b, 3, 875)
        batch_size = w.size(0)  # batch_size = b
        channel_size = w.size(1)
        time_steps = 7

        # k = w.view(batch_size * time_steps, w.size(1), int(w.size(2) / time_steps))    # one step

        w = w.view(batch_size, channel_size, int(w.size(2) / time_steps), time_steps)
        # segment ppg   shape:(b, c, 125, 7)
        w = w.transpose(1, 3)  # (b, 7, 125, c)
        w = w.transpose(2, 3)  # (b, 7, c, 125)
        w = w.contiguous().view(batch_size * time_steps, w.size(2), w.size(3))  # w ori:(b, 7, c, 125)   (b * 7, c, 125)

        w = self.conv1ecg(w)  # (b, 16, 875)
        w = self.bn1ecg(w)    #
        w = self.relu(w)
        w = self.layer1ecg(w)  # (b, 16, 875)
        w = self.layer2ecg(w)  # (b, 32, 438)
        w = self.layer3ecg(w)  # (b, 64, 219)
        w = self.avgpoolecg(w)  # (b, 64, 1)
        w = torch.squeeze(w)  # (b, 64)

        w = w.view(batch_size, time_steps, w.size(1))  # (b, 7, 64)
        w, (h0, b0) = self.lstm_ecg(w, self.hidden_cell)  # w:(5, 10, 64)   h0:(2, 5, 64) b0:(2, 5 ,64)
        w = h0[-1, :, :]  # (5, 64)

        x = self.transformerEncoder(w)  # (4, 5, 64)
        # x = torch.mean(x, dim=0)  # (5, 64)
        x = self.drop(x)
        x = self.fc(x)  # (5, 32)
        x = self.fc_final(x)  # (5, 2)

        return x


def resnet_lstm_mitbih(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


if __name__ == '__main__':
    model = resnet_lstm_mitbih(num_classes=2, block_name='BasicBlock')
    inputs = torch.rand(5, 3, 875)
    outputs = model(inputs)
    print(outputs.size())
