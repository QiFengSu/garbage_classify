#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：garbage_classify 
@File    ：excu_data.py
@IDE     ：PyCharm 
@Author  ：AC_sqf
@Date    ：2022/5/27 14:33 
'''
import os
import pandas as pd
def excu_data(path):
    datas = []
    labels = []
    files = os.listdir(path)
    for f in files:
        if f.endswith('txt'):
            d = open(os.path.join(path, f), 'r')
            data, label = d.read().split(', ')
            datas.append(data)
            labels.append(label)
            d.close()
    pd.DataFrame({'data':datas, 'label':labels}).to_csv('./datas.csv')

excu_data('./garbage_classify_v2/train_data_v2')