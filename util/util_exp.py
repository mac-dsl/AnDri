import numpy as np
######## updated 2026-10-08 ##########
import pandas as pd
from util.TSB_AD.metrics import metricor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import copy
import re
import pickle
import os

from util.util_andri import find_length, running_mean

import warnings
import random
import datetime

# from tqdm.notebook import tqdm
import time
import math
# from util.TranAD_base import *

# import tensorflow as tf
# import os
import sys
import argparse

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray', 
          'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan', 'pink', 'deepskyblue', 'lawngreen',
          'royalblue', 'darkgrey', 'darkorange', 'darkgreen','darkviolet','salmon','olivedrab','lightcoral','darkcyan','yellowgreen']
markers = ['o', 'x', '^', 'v', 's', '*', '+', '.', ',', '<', '>' , '1','2','3','4','p','h','H','D','d']
warnings.filterwarnings('ignore')

peak_columns =['AUC', 'Precision', 'Recall', 'F1', 'TH', 'RPrecision', 'RRecall', 'RF1', 'PaK']
peak_adj_columns = ['AUC', 'Precision', 'Recall', 'F1', 'TH', 'RPrecision', 'RRecall', 'RF1', 'PaK', 'F1_adj', 'Precision_adj', 'Recall_adj', 'roc_auc_adj']

########################################################################
# the below function is taken from OmniAnomaly code base directly
# https://github.com/NetManAIOps/OmniAnomaly
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    print(type(score), score.dtype, threshold)
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = metrics.roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc


###################################################################################################################
    
def peakf1_acc(label, score, th=0.5, plot_AUC=False, alpha=0.2):
    grader = metricor()
    result = pd.DataFrame(columns=peak_adj_columns)
    c = th
    if np.sum(label) != 0:
        auc = metrics.roc_auc_score(label, score)

        # plor ROC curve
        fpr, tpr, _ = metrics.roc_curve(label, score)
        pr, re, thresholds = metrics.precision_recall_curve(label, score)

        # print(f'LEN: {len(thresholds)}')
        if plot_AUC:
            dp = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            dp.plot()
        
        peak_f1, peak_ind = np.nanmax(2*(pr*re)/(pr+re)), np.nanargmax(2*(pr*re)/(pr+re))
        # print(peak_f1, peak_ind)


        peak_ths = thresholds[peak_ind] 
        # print(peak_ths)

        #range anomaly 
        preds = score > peak_ths
        Rrecall, ExistenceReward, OverlapReward = grader.range_recall_new(label, preds, alpha)
        Rprecision = grader.range_recall_new(preds, label, 0)[0]

        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)

        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))

        p_at_k = np.where(score > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k

        ## Adjustment csae
        pred_adj = adjust_predicts(score, label,
                threshold=peak_ths,
                pred=None,
                calc_latency=False)
        
        f1_adj, pr_adj, re_adj, _, _, _, _, auc_adj = calc_point2point(pred_adj, label)

        
        result.loc[0] = [auc, pr[peak_ind], re[peak_ind], peak_f1, peak_ths, Rprecision, Rrecall, Rf, precision_at_k, f1_adj, pr_adj, re_adj, auc_adj]
        return result

def result_f1_acc(methods, scores, label, th=0.5):
    result_org = pd.DataFrame(columns=['method'] + peak_adj_columns)
    j = 0
    for i, method in enumerate(methods):
        # r_tmp = get_acc(label.reshape(-1)[:len(scores[i])], np.array(scores[i]), slidingWindow, ths)
        print('LEN SCORES:', len(scores[i]), len(label))
        if len(scores[i]) < len(label):
            label_rev = label[len(label)-len(scores[i]):].copy()
            sc_t = scores[i]
        elif len(scores[i]) > len(label) + 100:
            sc_t = scores[i][len(scores[i])-len(label):]
            label_rev = label
        else:
            label_rev = label
            sc_t = scores[i]
        print('LEN SCORES:', len(sc_t), len(label), len(label_rev))
        r_tmp = peakf1_acc(label_rev.reshape(-1)[:len(sc_t)], np.array(sc_t), th=th, plot_AUC=False)
        if r_tmp is not None:
            result_org.loc[j] = [method] + list(r_tmp.loc[0])
        else:
            result_org.loc[j] = [method] + [0]*len(peak_adj_columns)
        j+=1

    # display(result_org)
    return result_org


def save_pickle(filename, var):   
    with open(filename, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        var = pickle.load(f)
    return var

###################################################################################################################
######## updated 2026-10-08 ##########
def get_data(data_name):
    """Read the saved univariate inputs without changing labels or selecting stations."""
    from pathlib import Path
    ######## updated 2026-10-09 ##########
    folder = Path(__file__).resolve().parents[1] / 'data' / 'processed' / data_name
    files = sorted(folder.glob('*_processed.csv' if data_name == 'climate' else '*.csv'))
    data_list, label_list, filelist = [], [], []
    for path in files:
        if data_name == 'climate':
            df = pd.read_csv(path, usecols=['data', 'label'])
            data, label = df['data'], df['label']
        elif data_name == 'traffic':
            df = pd.read_csv(path, usecols=['total_flow', 'total_flow_label'])
            data, label = df['total_flow'], df['total_flow_label']
        else:
            df = pd.read_csv(path, usecols=['Data', 'Label'])
            data, label = df['Data'], df['Label']
        data_list.append(data.to_numpy(dtype=float))
        label_list.append(label.to_numpy(dtype=int))
        filelist.append(path.name)
    return data_list, label_list, filelist


######## updated 2026-10-08 ##########
def get_multi_data(data_name):
    """Read each saved feature with its existing label and training boundary."""
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / 'data' / 'processed' / data_name
    data_list, label_list, filelist = [], [], []
    for path in sorted(folder.glob('*.csv')):
        df = pd.read_csv(path)
        label = df['label'].to_numpy(dtype=int)
        for column in df.columns.drop('label'):
            data_list.append(df[column].to_numpy(dtype=float))
            label_list.append(label)
            filelist.append(f'{path.name}__{column}')
    return data_list, label_list, filelist
