# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/10/9 15:07
@Author  : Rao Zhi
@File    : ppg_derivative.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch


def use_derivative(x_input, fs):
    """
    X_input: (None, 875, 1)
    fs     : 125
    """
    dt1 = (x_input[:, :, 1:] - x_input[:, :, :-1]) * fs     # (None, 874, 1)  ———pad———> (None, 875, 1)
    dt2 = (dt1[:, :, 1:] - dt1[:, :, :-1]) * fs             # (None, 873, 1)  ———pad———> (None, 875, 1)

    # under padding
    padded_dt1 = torch.nn.functional.pad(dt1, (0, 1, 0, 0), value=0)
    padded_dt2 = torch.nn.functional.pad(dt2, (0, 2, 0, 0), value=0)
    # (0, 0, 0, 1) Indicates the fill size of the left, right, top and bottom edges, respectively

    x = torch.cat([x_input, padded_dt1, padded_dt2], dim=1)
    return x


if __name__ == '__main__':
    inputs = torch.rand(1024, 1, 875)
    fs = 125
    y = use_derivative(inputs, fs)
    print()
