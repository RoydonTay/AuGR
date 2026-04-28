#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5
# @Author  : hongji.lai (hongji.lai@shopee.com)
# @Software: PyCharm
# @File    : auc.py

import numpy as np
from sklearn.metrics import roc_auc_score


# AUC
def safe_auc(y_true, y_score, k=0, ignore_ties=False):
    if np.sum(y_true) == len(y_true) or np.sum(y_true) == 0:
        return np.nan
    return roc_auc_score(y_true, y_score)
