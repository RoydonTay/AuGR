#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5
# @Author  : hongji.lai (hongji.lai@shopee.com)
# @Software: PyCharm
# @File    : mrr.py

import numpy as np


# MRR
def safe_mrr(y_true, y_score, k, ignore_ties):
    if np.sum(y_true) == 0:
        return np.nan
    zip_list = sorted(list(zip(y_true, y_score)), key=lambda x: x[1], reverse=True)
    mrr = [
        1 / (float(index) + 1)
        for index, click in enumerate([i[0] for i in zip_list])
        if click == 1
    ]
    return np.sum(mrr) / np.sum(y_true)
