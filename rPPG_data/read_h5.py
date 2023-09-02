# -*- coding: utf-8 -*-
"""
@Project ：My_BP_code 
@Time    : 2023/8/29 16:24
@Author  : Rao Zhi
@File    : read_h5.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import h5py
import matplotlib.pyplot as plt

h5_file = 'rPPG_bp_subject.h5'

with h5py.File(h5_file, 'r') as f:
    # load rPPG and BP data as well as the subject numbers the samples belong to

    rPPG = f.get('/rPPG')
    BP = f.get('/label')
    subject_idx = f.get('/subject_idx')

    print(rPPG[1, :].shape)

    plt.plot(rPPG[1, :])
    plt.show()

    print(BP.shape)
    print(subject_idx.shape)



