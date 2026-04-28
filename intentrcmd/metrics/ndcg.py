#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5
# @Author  : hongji.lai (hongji.lai@shopee.com)
# @Software: PyCharm
# @File    : ndcg.py

import numpy as np
from sklearn.metrics import ndcg_score


# NDCG
def safe_ndcg(y_true, y_score, k, ignore_ties):
    if (k==1 and len(y_true)==1) or len(y_true) < k or np.sum(y_true) == 0:
        return np.nan
    return ndcg_score(y_true=[y_true], y_score=[y_score], k=k, ignore_ties=ignore_ties)
