# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/13 16:49
@Author  : Rao Zhi
@File    : pytorch_fine-tune.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        pass


model = Model()

for name, param in model.named_parameters():
    print(name, param.shape)

# method one
# Freeze "conv1", "conv2" : Each layer contains two parameters (weight, bias)
# So the first two layers are the first four parameters

model_param = model.parameters()  # model_param is a generator
model_param_list = list(model_param)    # make the generator become a list

a0 = model_param_list[0]
a1 = model_param_list[1]
a2 = model_param_list[2]
a3 = model_param_list[3]
a4 = model_param_list[4]
a5 = model_param_list[5]

# for param in model.parameters()[:4]:    # TypeError: 'generator' object is not subscriptable(索引到)
for param in list(model.parameters())[:4]:
    param.requires_grad = False

# method two
# Freeze based on the name of the parameter layer
freeze_layers = ("conv1", "conv2")
for name, param in model.named_parameters():
    print(name, param.shape)
    if name.split(".")[0] in freeze_layers:
        param.requires_grad = False
