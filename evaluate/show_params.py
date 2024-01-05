# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2024/1/4 10:00
@Author  : Rao Zhi
@File    : show_params.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 
@描述     :对比模型的参数量
@detail   ：Compare the number of parameters in the model
@infer    :
"""

import torch
import torch.nn as nn
from model.bpnet_cvprw import resnet50
# from model.MSR_tranformer_bp_v7 import msr_tf_bp  # msr_tf_bp_mse_data_split+derivative_2023121802
from model.MSR_tranformer_bp import msr_tf_bp  # msr_tf_bp_mse_data_split+derivative_2023121802

# 假设你有一个保存的模型文件
# model_name = 'cvprw_split_data_1_channel'
model_name = 'msr_tf_bp_mse_data_split+derivative_2023121802'
# saved_model_path = 'logs/' + model_name + '/best_w.pth'

input_channel = 3
model_type = "msr_tf_bp"  # or msr_tf_bp  cvprw

if model_type == 'cvprw':
    model = resnet50(input_c=input_channel, num_classes=2)  # cvprw
elif model_type == 'msr_tf_bp':
    model = msr_tf_bp(input_channel=input_channel, layers=[1, 1, 1, 1], num_classes=2)  # ours
else:
    pass


model.load_state_dict(torch.load('../logs/' + model_name + '/best_w.pth')['state_dict'])

state_dict = model.state_dict()

# 计算总参数量
total_params = sum(p.numel() for p in state_dict.values())
print(f"Total parameters in the model: {total_params}")

