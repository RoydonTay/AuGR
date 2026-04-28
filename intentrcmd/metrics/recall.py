#! /usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/5
# @Author  : hongji.lai (hongji.lai@shopee.com)
# @Software: PyCharm
# @File    : recall.py

import numpy as np


# Recall
def safe_recall(y_true, y_score, k, ignore_ties):
    if not ignore_ties:
        y_score_ = np.unique(y_score)
    else:
        y_score_ = y_score
    k = np.minimum(k, len(y_score_))
    ind_top_k = np.argpartition(y_score_, -k)[-k]
    score_top_k = y_score_[ind_top_k]

    recall = [y_true[i] for i in range(len(y_score)) if y_score[i] >= score_top_k]
    if np.sum(y_true) > 0:
        return np.sum(recall) / np.sum(y_true)
    return np.nan
