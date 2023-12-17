# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/12/16 17:37
@Author  : Rao Zhi
@File    : inference.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch
import warnings
import matplotlib.pyplot as plt
from model.Resnet import resnet18

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ppgs = torch.load("ppg.h5")
labels = torch.load("label.h5")

# out_domain infer
ppgs = torch.load("../data_normal/train/ppg.h5")

model = resnet18(input_c=1, num_classes=1)
model = model.to(device)

model_name = 'abnormal_pulse_detection_total=1798_2023121711'
model.load_state_dict(torch.load('save/' + model_name + '/best_w.pth')['state_dict'])
best_epoch = torch.load('save/' + model_name + '/best_w.pth')["epoch"]
print(f"best epoch:{best_epoch}")

model.eval()
# ppg = ppg[1200]
# label = labels[1200]
pre = []
inference_head = 20000
with torch.no_grad():
    for i in range(100):
        # ppg = ppgs[i + 1200].unsqueeze(dim=0).unsqueeze(dim=0)
        ppg = ppgs[inference_head + i].unsqueeze(dim=0).unsqueeze(dim=0)
        ppg = ppg.to(device)

        a = model(ppg).cpu().squeeze()
        output = torch.sigmoid(model(ppg).cpu().squeeze())
        predict = (output >= 0.5).float()  # 二分类阈值为0.5
        pre.append(predict)
        plt.plot(ppg.squeeze().cpu())
        # plt.title(f"predict:{predict}, gt:{labels[inference_head + i]}")
        plt.title(f"predict:{predict}")
        # plt.title(f"predict:{predict}, gt:{labels[i]}")
        plt.show()

# print(pre.count(1))   # show count  of one in list


