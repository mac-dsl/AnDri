import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.TSB_AD.metrics import metricor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import copy
import re
import pickle
import os

from util.plot_andri import find_anomaly_intervals
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
def get_climate_label(df):
    w_anom_lists = ['Thunderstorms', 'Heavy Rain', 'Heavy Snow']
    df.loc[df['Precip. Amount (mm)'] > 100, 'Precip. Amount (mm)'] = 100
    df['datetime'] = pd.to_datetime(df['Date/Time (LST)'], format='%Y-%m-%d %H:%M')
    df['date'] = df['datetime'].dt.date
    df = df.sort_values('datetime')

    df['precip_24h'] = (df.rolling('24h', on='datetime')['Precip. Amount (mm)'].sum())
    # dailiy_precip = (df.groupby('date', as_index=False)['Precip. Amount (mm)'].sum())
    # df = df.merge(dailiy_precip, on='date', how='left', suffixes=('', '_daily'))

    df['heavy'] = 0
    df.loc[df['Precip. Amount (mm)'] > 4, 'heavy'] = 1
    df.loc[df['precip_24h'] >= 10, 'heavy'] = 1

    for i, tw in enumerate(w_anom_lists):
        mask = df['Weather'].str.contains(tw, na=False)
        df.loc[mask, 'heavy'] = 1

    df['Precip. Amount (mm)'] = df['Precip. Amount (mm)'].interpolate()

    return df

def shift_time(sel_time, sel_len):
    total_month = sel_time.year * 12 + sel_time.month - sel_len * 6
    
    year = total_month // 12
    month = total_month % 12
    
    if month == 0:
        year -= 1
        month = 12

    return datetime.datetime(year, month, sel_time.day)

def get_data(data_name, target=None, anom_lists=None, short=True, traffic_sel='total_flow', w_range=4, y_weight=0.5, w_weight=0.5):
    data_list, label_list = [], []
    if data_name == 'INN_Sensor':
        dir_t = './data/processed/real iot/'
        filelist_test = os.listdir(dir_t)
        filelist_test.sort()

    elif data_name == 'climate':
        dir_t = './data/processed/2021_2025_precip_selected/'
        filelist_test_t = os.listdir(dir_t)
        filelist_test = [f for f in filelist_test_t if f.endswith('processed.csv')]
        filelist_test.sort()
        provinces, stations = [], []

    elif data_name == 'traffic':
        df_stat = pd.read_csv('/home/parkj182/research/PeMS/stat_missing_station.csv', index_col=None)
        stationIDs = df_stat[(df_stat['all_len']>=149396) & (df_stat['missing_hour']<=103)]['stationID'].values
        dir_t = f'./data/processed/PeMS/101_N/drifts/'
        filelist_test_all = os.listdir(dir_t)
        end_text = f'th_{w_range}_y_{y_weight}_w_{w_weight}'
        start_text = f'rev_{end_text}'
        
        filelist_test_t = [f for f in filelist_test_all if f.startswith(start_text)]
        filelist_th_t = [f for f in filelist_test_all if f.endswith(f'{end_text}.csv')]

        print('LEN:', len(filelist_test_t), len(filelist_th_t), f'{end_text}.csv')

        pattern = re.compile(r".*N_(\d+)_DST")
        filelist_test = [
            f for f in filelist_test_t
            if (m := pattern.match(f)) and int(m.group(1)) in stationIDs
        ]
        filelist_th = [
            f for f in filelist_th_t
            if (m := pattern.match(f)) and int(m.group(1)) in stationIDs
        ]
        filelist_test.sort()
        # filelist_th.sort()

    for f in filelist_test:
        # print(f)
        if data_name in ['SensorScope', 'NAB', 'Occupancy']:
            # if data_name == 'Occupancy' and f.split('-')[2][0] == 0:
                # continue
            if data_name == 'NAB' and f.split('_')[2][:3] == 'art':
                continue
            df_t = pd.read_csv(f'{dir_t}{f}', index_col=None, header=None)

        else:
            df_t = pd.read_csv(f'{dir_t}{f}', index_col=None)

        if 'level_0' in df_t.columns:
            df_t = df_t.drop(columns=['level_0'])
        if data_name == 'INN_Sensor':
            df_t['anomaly'] = df_t['anomaly_pattern'] | df_t['anomaly_point']
            data, label = df_t['value'].to_numpy(), df_t['anomaly'].to_numpy()
            data = data.astype(float)
        elif data_name == 'traffic':
            df_t = pd.read_csv(f'{dir_t}{f}', index_col=None)
            print(f)
            data, label = df_t[traffic_sel].to_numpy(), df_t[f'{traffic_sel}_label'].to_numpy()
        elif data_name == 'climate':
            provinces.append(f.split('_')[0])
            stations.append(f.split('_')[1])
            df_t = get_climate_label(df_t)
            init_date = datetime.datetime(2023, 7, 1)
            start_date = shift_time(init_date, int(short))
            df_t = df_t[df_t['datetime']>=start_date]
            data, label = df_t['Precip. Amount (mm)'].to_numpy(), df_t['heavy'].to_numpy()
            
            data = data.astype(float)
        elif data_name in ['SensorScope', 'NAB', 'Occupancy']:
            data, label = df_t.iloc[:,0].to_numpy(), df_t.iloc[:,1].to_numpy()
        else:
            data, label = df_t['Data'].to_numpy(), df_t['Label'].to_numpy()
            data = data.astype(float)


        data_list.append(data)
        label_list.append(label)
            
    if data_name == 'climate':
        print('climate_data:', len(data_list[0]))
        return data_list, label_list, filelist_test, provinces, stations
    elif data_name == 'traffic':
        print('traffic_data:', len(data_list[0]))
        return data_list, label_list, filelist_test, filelist_th, stationIDs
    else:
        return data_list, label_list, filelist_test

### Read all dimensions and run each dimension for AnDri
def get_multi_data(data_name):
    data_list, label_list = [], []

    if data_name in ['SMD']:
        ## Applied dimension reductions
        dir_t = './data/processed/SMD/'
        filelist_test = []
        filelist = os.listdir(dir_t)
        filelist_csv = [file for file in filelist if file.endswith('subset.csv')]

        sel_list = [f for f in filelist_csv if f.split('-')[0] == 'machine']

        for f in sel_list:
            df = pd.read_csv(f'{dir_t}{f}')
            # data, label = df.iloc[:,sel_columns].astype('float64').to_numpy(), df.iloc[:,-1].astype('float64').to_numpy()
            label = df['label'].astype('float64').to_numpy()
            for j in range(len(df.columns)-1):
                filelist_test.append(f'{j}_{f}')
                data_list.append(df.iloc[:,j].astype('float64').to_numpy())
                label_list.append(label)

    elif data_name == 'SWaT':
        dir = './data/processed/SWaT/'
        # f = 'SWaT_dataset_Jul_19_v2.csv'
        f = 'SWaT_processed.csv'
        df = pd.read_csv(f'{dir}{f}', index_col=None)
        label = df['label'].to_numpy()

        filelist_test = []
        for j in range(1, len(df.columns)-1):
            data_list.append(df.iloc[:,j].to_numpy())
            label_list.append(label)
            filelist_test.append(f'{j}_{f}')

    elif data_name == 'WADI':
        filelist_test = []
        dir = '/home/parkj182/research/WADI/'
        filelist = os.listdir(dir)
        filelist_csv = [file for file in filelist if file.endswith('processed.csv')]
        for f in filelist_csv:
            df = pd.read_csv(f'{dir}{f}')
            label = df.iloc[:,-1].astype('float64').to_numpy()
            for j in range(len(df.columns)-1):
                filelist_test.append(f'{j}_{f}')
                data_list.append(df.iloc[:,j].astype('float64').to_numpy())
                label_list.append(label)


    return data_list, label_list, filelist_test