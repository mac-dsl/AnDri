import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import copy
import datetime


# from util.plot_andri import find_anomaly_intervals, plot_anomaly, plotFigRev, plot_membership, plot_cluster_color
# from util.util_andri import find_length, divide_subseq, norm_seq
from util.util_exp import result_acc, result_f1_acc, save_pickle, load_pickle
from util.util_data import load_score_clf
import re
import warnings
import random


import requests
from tqdm.notebook import tqdm


import time
import math
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray', 
          'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan', 'pink', 'deepskyblue', 'lawngreen',
          'royalblue', 'darkgrey', 'darkorange', 'darkgreen','darkviolet','salmon','olivedrab','lightcoral','darkcyan','yellowgreen']
markers = ['o', 'x', '^', 'v', 's', '*', '+', '.', ',', '<', '>' , '1','2','3','4','p','h','H','D','d']
warnings.filterwarnings('ignore')

param_types = {
    'l': int,
    'k': int,
    'nm': int,
    'Wmax': int,
    'delta': int,
    'anom': int,
    'rmin': float,
    'd': str,
    'linkage': str,
    'rollback':bool,
}

def get_andri_param(fname):
    pattern = r'_(' + '|'.join(param_types.keys()) + r')_([a-zA-Z0-9.\-]+)'
    matches = re.findall(pattern, fname)

    params = {}
    for k, v in matches:
        type_func = param_types[k]
        params[k] = type_func(v)
    # print(params)
    return params

def running_mean(x,N):
	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N

def load_score_clf_smd(dir, sel_data, method, conditions, machine1=None, machine2=None):
    norma_clf = False
    filelists = os.listdir(f'{dir}{sel_data}/')
    print('Load score', method)
    
    f_lists = [f for f in filelists if f.startswith(method)]
    f_list_score = [f for f in f_lists if f.endswith('scores.pickle')]
    f_list_score.sort()
    print(f_list_score)

    if 'AnDri' in method:
        f_list_off = [f for f in f_lists if f.endswith('off_.pickle')]
        f_list_off.sort()
        f_list_on = [f for f in f_lists if f.endswith('on_.pickle')]
        f_list_on.sort()


    elif method in ['NormA', 'SAND']:
        f_list_off = [f for f in f_lists if f.endswith('clf_.pickle')]
        f_list_off.sort()
        print('NormA or SAND', f_list_off)
        
        if len(f_list_off) == 0:
            norma_clf = True
            print('CLF: ', norma_clf)

    print('ID:', machine1)
    
    if sel_data in ['INN_Sensor']:
        machine1_s = [f for f in f_list_score if int(f.split('_')[2].split('.')[0][-1]) == machine1]
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = [f for f in f_list_off if int(f.split('_')[2].split('.')[0][-1]) == machine1]
            machine1_off.sort()
            if method == 'AnDri':
                machine1_on = [f for f in f_list_on if int(f.split('_')[2].split('.')[0][-1]) == machine1]
                machine1_on.sort()
    elif sel_data in ['SensorScope']:
        machine1_s = [f for f in f_list_score if int(f.split('_')[1].split('-')[1]) == machine1]
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = [f for f in f_list_off if int(f.split('_')[1].split('-')[1]) == machine1]
            machine1_off.sort()
            if method == 'AnDri':
                machine1_on = [f for f in f_list_on if int(f.split('_')[1].split('-')[1]) == machine1]
                machine1_on.sort()

    elif sel_data in ['NAB']:
        machine1_s = [f for f in f_list_score if f.split('_')[3] == machine1]
        machine2_s = [f for f in machine1_s if int(f.split('_')[4]) == machine2]
        print('NAB:', f_list_score[0], machine2_s, len(machine1_s))
        print('ID2: ', machine2)
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = [f for f in f_list_off if f.split('_')[3] == machine1]
            machine2_off = [f for f in machine1_off if int(f.split('_')[4]) == machine2]
            machine2_off.sort()
            if method == 'AnDri':
                machine1_on = [f for f in f_list_on if f.split('_')[3] == machine1]
                machine2_on = [f for f in machine1_on if int(f.split('_')[4]) == machine2]
                machine2_on.sort()

    elif sel_data in ['SMD']:
        div_idx = 2 if method in ['AnDri', 'NormA', 'SAND', 'CABD'] else 1
        # print(div_idx, f_list_score[0].split('_')[2])
        # print(f_list_score)
        machine1_s = [f for f in f_list_score if int(f.split('_')[div_idx].split('-')[1]) == machine1]
        machine2_s = [f for f in machine1_s if int(f.split('_')[div_idx].split('-')[2]) == machine2]
        # print(machine1_s)
        # print(machine2_s)
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = [f for f in f_list_off if int(f.split('_')[div_idx].split('-')[1]) == machine1]
            machine2_off = [f for f in machine1_off if int(f.split('_')[div_idx].split('-')[2]) == machine2]
            machine2_off.sort()
            if method == 'AnDri':
                machine1_on = [f for f in f_list_on if int(f.split('_')[div_idx].split('-')[1]) == machine1]
                machine2_on = [f for f in machine1_on if int(f.split('_')[2].split('-')[2]) == machine2]
                machine2_on.sort()
    elif sel_data in ['WADI']:
        div_idx = 2 if method in ['AnDri', 'NormA', 'SAND', 'CABD'] else 1
        machine1_s = [f for f in f_list_score if int(f.split('_')[div_idx]) == machine1]
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = [f for f in f_list_off if int(f.split('_')[2]) == machine1]
            machine1_off.sort()
            if method == 'AnDri':
                machine1_on = [f for f in f_list_on if int(f.split('_')[2]) == machine1]
                machine1_on.sort()
    elif sel_data in ['SWaT']:
        machine1_s = f_list_score
        if method in ['AnDri', 'NormA', 'SAND']:
            machine1_off = f_list_off
            machine1_off.sort()
            if method == 'AnDri':
                machine1_on = f_list_on
                machine1_on.sort()

    if sel_data in ['SMD', 'NAB']: 
        machine_s = machine2_s
    else:
        machine_s = machine1_s

    machine_s.sort()
    print('SEL:', machine_s)

    scores_all, clfs_all, dims = [], [], []

    if method in ['AnDri']:
        print('Conditions :', conditions)

        if sel_data in ['SMD', 'NAB']:
            machine_off, machine_on = machine2_off, machine2_on
        else:
            machine_off, machine_on = machine1_off, machine1_on

        for sub_s, sub_off, sub_on in zip(machine_s, machine_off, machine_on):
            params = get_andri_param(sub_s)
            print('where', sub_s)
            if (conditions['d'] == params['d'] and conditions['k'] == params['k'] and conditions['linkage'] == params['linkage'] and conditions['Wmax'] == params['Wmax'] and conditions['delta'] == params['delta'] and conditions['rmin'] == params['rmin']):
                clf = []
                score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                score = np.array(score, dtype=float)
                clf.append(load_pickle(f'{dir}{sel_data}/{sub_off}'))
                clf.append(load_pickle(f'{dir}{sel_data}/{sub_on}'))
                scores_all.append(score)
                clfs_all.append(clf)
                if sel_data in ['SMD', 'SWaT', 'WADI']:
                    dim_num = int(sub_s.split('_')[1])
                    dims.append(dim_num)

    elif method in ['NormA', 'SAND']:
        if norma_clf:
            for sub_s in machine_s:
                if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                    score = np.array(load_pickle(f'{dir}{sel_data}/{sub_s}'), dtype=float)
                    scores_all.append(score)
                    if sel_data in ['SMD', 'SWaT', 'WADI']:
                        dim_num = int(sub_s.split('_')[1])
                        dims.append(dim_num)
                    print(sub_s)

        else:
            for sub_s, sub_off in zip(machine_s, machine_off):
                if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                    score = np.array(load_pickle(f'{dir}{sel_data}/{sub_s}'), dtype=float)
                    scores_all.append(score)
                    clfs_all.append(load_pickle(f'{dir}{sel_data}/{sub_off}'))
                    if sel_data in ['SMD', 'SWaT', 'WADI']:
                        dim_num = int(sub_s.split('_')[1])
                        dims.append(dim_num)
    else:
        for sub_s in machine_s:
            score = np.array(load_pickle(f'{dir}{sel_data}/{sub_s}'), dtype=float)
            scores_all.append(score)
            if sel_data in ['SMD', 'SWaT', 'WADI'] and method == 'CABD':
                dim_num = int(sub_s.split('_')[1])
                dims.append(dim_num)

    return scores_all, clfs_all, dims

def recompute_score_by_thresholds_smd(method, save_dir, score_dir, save_to_dir, param):
    # save_dir = './2021_2025_precip_filtered'
    
    filelist_t = os.listdir(save_dir)
    # print(filelist_t)
    data_name = save_dir.split('/')[-1]
    print(data_name)
    
    if data_name == '':
        data_name = 'INN_Sensor'
        filelist = [f for f in filelist_t if f.endswith('.csv')]
        machine2 = None
    if data_name == 'SMD':
        filelist = [f for f in filelist_t if f.endswith('subset.csv')]
    elif data_name in ['SWaT', 'WADI']:
        filelist = [f for f in filelist_t if f.endswith('processed.csv')]
        machine2 = None
    elif data_name in ['SensorScope', 'NAB', 'Occupancy']:
        filelist = [f for f in filelist_t if f.endswith('.out')]
        filelist.sort()

    print(filelist)

    if method == 'AnDri':
        method_to = ['AnDri (off)', 'AnDri (on)']
        normalize = param['d']
        k=param['k']
        linkage = param['linkage']
        Wmax = param['Wmax']
        delta = param['delta']
        rmin= param['rmin']
    elif method in ['NormA', 'SAND']:
        method_to = [method]
        normalize = param['d']
    else:
        method_to = [method]


    results = pd.DataFrame()

    if data_name == 'SMD':
        df_sand_machine= pd.read_csv(f'{score_dir}/SMD_all_SAND_machine.csv', index_col=None)


    for f in filelist:
        if data_name == 'SMD':
            machine1 = int(f.split('-')[1])
            machine2 = int(f.split('-')[2].split('_')[0])

            df_chk = df_sand_machine[df_sand_machine['machine1'] == machine1]
            df_chk2 = df_chk[df_chk['machine2'] == machine2]

            if len(df_chk2) == 0:
                print(f'No SAND score for {machine1}-{machine2}, skip')
                continue 
        elif data_name in ['WADI']:
            machine1 = int(f.split('_')[0])
        elif data_name in ['INN_Sensor']:
            machine1 = int(f.split(' ')[2].split('.')[0])
        elif data_name in ['SensorScope']:
            print(f)
            machine1 = int(f.split('-')[1].split('.')[0])
            machine2 = None
        elif data_name in ['NAB']:
            machine1 = f.split('_')[2]
            machine2 = int(f.split('_')[3].split('.')[0])
            print(f, machine1, machine2)


        if method == 'AnDri':
            conditions = {'d': normalize, 'k':k, 'linkage':linkage,'Wmax':Wmax, 'delta':delta, 'rmin':rmin}
        elif method in ['NormA', 'SAND']:
            conditions = {'d': param['d']}
        else:
            conditions = {}

        print(f)
        df_ID = pd.read_csv(f'{save_dir}/{f}')

        ## load scores and models
        scores_all, clf_all, dims = load_score_clf_smd(score_dir, data_name, method, conditions, machine1, machine2)
        print(f'For {machine1}-{machine2}, loaded dims: {dims}, {len(dims)}')
        print(len(df_ID.columns), len(scores_all))
        if len(df_ID.columns) == len(scores_all):
            idx = dims.index(max(dims))
            print('SMD:', len(dims), max(dims), dims[idx], idx)
            del scores_all[idx]
            del clf_all[idx]
            del dims[idx]
        # print(len(scores_all), len(clf_all))

        if method in ['AnDri', 'NormA', 'SAND', 'CABD'] and len(dims) == 0: 
            if data_name not in ['INN_Sensor', 'SensorScope', 'NAB', 'Occupancy']:
                continue

        if method == 'AnDri':
            score_off, score_on = [], []
            if data_name in ['SMD', 'SWaT', 'WADI']:
                for score, clf, dim in zip(scores_all, clf_all, dims):
                    # print(dim, len(clf), clf[0].scores.shape, clf[0].scores_rev.shape)
                    # score_off.append(clf[0].scores_rev)
                    # score_on.append(clf[1].scores_rev)
                    # print('chk score', len(score))
                    score_off.append(score[0])
                    score_on.append(score[1])
            elif data_name in ['INN_Sensor', 'SensorScope', 'NAB', 'Occupancy']:
                for score, clf in zip(scores_all, clf_all):
                    # print('chk score', len(score))
                    score_off.append(score[0])
                    score_on.append(score[1])

            score = []
            if data_name in ['SMD', 'SWaT', 'WADI']:
                score.append(np.mean(score_off, axis=0))
                score.append(np.mean(score_on, axis=0))
            elif data_name in ['INN_Sensor', 'SensorScope', 'NAB', 'Occupancy']:
                score.append(score_off[0][:len(df_ID)])
                score.append(score_on[0][:len(df_ID)])

        elif method in ['NormA', 'SAND', 'CABD']:
            if data_name in ['SMD', 'SWaT', 'WADI']:
                score = np.mean(scores_all, axis=0)
            elif data_name in ['INN_Sensor']:
                score = scores_all[0]
            elif data_name in ['SensorScope', 'NAB', 'Occupancy']:
                print('NormA SAND', len(scores_all), len(scores_all[0]) ) #, len(scores_all[0][0]))
                score = [scores_all[0][0][:len(df_ID)]]

        else:
            print(len(scores_all), len(scores_all[0]))
            score = [scores_all[0][0][:len(df_ID)]]

        ## Data
        
        if data_name == 'INN_Sensor':
            df_ID['anomaly'] = df_ID['anomaly_pattern'] | df_ID['anomaly_point']
            # print('LEN:', len(score), len(df_ID['anomaly'].to_numpy()))
            # print('score?', len(score[0]), len(score[1]))
            result_t = result_f1_acc(method_to, score, df_ID['anomaly'].to_numpy())
        elif data_name in ['SensorScope', 'NAB', 'Occupancy']:
            print(f'LEN: {len(df_ID)}, score={len(score)}, {len(score[0])}')
            result_t = result_f1_acc(method_to, score, df_ID.iloc[:,1].to_numpy())
        else:
            result_t = result_f1_acc(method_to, score, df_ID['label'].to_numpy())
        result_t['machine1'] = machine1
        result_t['machine2'] = machine2
        results = pd.concat([results, result_t])

    # if sel_rev and method == 'AnDri':
        # if sel_param is not None:
            # results.to_csv(f'{save_to_dir}/revised_thresholds_{method}_{sel_param}_rev.csv', index=None)
    # else:
    results.to_csv(f'{save_to_dir}/revised_thresholds_{data_name}_all_{method}.csv', index=None)
    print(f'{save_to_dir}/revised_thresholds_{data_name}_all_{method}.csv')
    return results


def draw_selected_station_smd(save_dir, score_dir, machine1, machine2, methods = 'all', dim=None):

    filelist_t = os.listdir(save_dir)
    # print(filelist_t)
    data_name = save_dir.split('/')[-2]
    print(data_name)
    if data_name == '':
        data_name = 'INN_Sensor'
        filelist = [f for f in filelist_t if f.endswith('.csv')]
        machine2 = None
    if data_name == 'SMD':
        filelist = [f for f in filelist_t if f.endswith('subset.csv')]
    elif data_name in ['SWaT', 'WADI']:
        filelist = [f for f in filelist_t if f.endswith('processed.csv')]
        machine2 = None

    # filelist = [f for f in filelist_t if f.endswith('subset.csv')]

    if methods == 'all':
        methods = ['AnDri', 'CABD', 'NormA', 'SAND', 'TranAD', 'ARCUS', 'DIVAD']
    else:
        methods = methods

    scores, slabels = [], []
    clfs = []
    results = pd.DataFrame()
    df_ID = pd.DataFrame()
    for method in methods:
        if method == 'AnDri':
            method_to = ['AnDri (off)', 'AnDri (on)']
            normalize = 'zero-mean'
            k=1
            linkage = 'ward'
            Wmax = 20
            delta = 10
            rmin= 0.02
        elif method in ['NormA', 'SAND']:
            normalize = 'zero-mean'
            method_to = [method]
        else:
            method_to = [method]
        print(filelist)
        sel_m1 = [f for f in filelist if int(f.split('-')[1]) == machine1]
        sel_IDs = [f for f in sel_m1 if int(f.split('-')[2].split('_')[0]) == machine2]
        
        sel_IDs.sort()
        print(sel_IDs)

        if len(df_ID) == 0:
            ## Data
            df_ID = pd.read_csv(f'{save_dir}/{sel_IDs[0]}')        

            # data = df_ID['Precip. Amount (mm)'].to_numpy()
            label = df_ID['label'].to_numpy()

        if method == 'AnDri':
            conditions = {'d': normalize, 'k':k, 'linkage':'ward','Wmax':Wmax, 'delta':delta, 'rmin':rmin}
        elif method in ['NormA', 'SAND']:
            conditions = {'d': normalize}
        else:
            conditions = {}

        ## load scores and models
        scores_all, clf_all, dims = load_score_clf_smd(score_dir, 'SMD', method, conditions, machine1, machine2)
        # print(len(scores_all), len(clf_all))
        print(f'For {machine1}-{machine2}, loaded dims: {dims}, {len(dims)}')
        if len(df_ID.columns) == len(scores_all):
            idx = dims.index(max(dims))
            print('SMD:', len(dims), max(dims), dims[idx], idx)
            del scores_all[idx]
            del clf_all[idx]
            del dims[idx]

        if method == 'AnDri':
            score_off, score_on = [], []
            for score, clf in zip(scores_all, clf_all):
                try:
                    # print('num clf', len(clf))
                    if len(clf[1].scores_rev) < len(clf[0].scores_rev):
                        clf[1].scores_rev = np.pad(clf[1].scores_rev, (0, len(clf[0].scores_rev)-len(clf[1].scores_rev)), mode='constant')
                    score_on.append(np.array(clf[1].scores_rev, dtype=float))
                    score_off.append(np.array(clf[0].scores_rev, dtype=float))
                except:
                    score_off.append(score[0])
                    score_on.append(score[1])


            score = []
            print('off: ', len(score_off), 'on:', len(score_on))
            # for i, s_off in enumerate(score_off):
                # print(f'off ind {i} has {len(s_off)}, {s_off.dtype}')

            # for i, s_on in enumerate(score_on):
                # print(f'on ind {i} has {len(s_on)}, {s_on.dtype}')
            if dim is None:
                scores.append(np.mean(score_off, axis=0))
                scores.append(np.mean(score_on, axis=0))
            else:
                scores.append(score_off[dim])
                scores.append(score_on[dim])

            clfs.append(clf[0])
            clfs.append(clf[1])
            slabels.append('AnDri (off)')
            slabels.append('AnDri (on)')
        elif method in ['NormA', 'SAND', 'CABD']:
            if data_name in ['SMD', 'SWaT', 'WADI']:
                for i, sc in enumerate(scores_all):
                    print(f'{i} for len: {len(sc[0])} and type: {np.array(sc[0]).dtype}')
                sc_all = []
                for i, sc in enumerate(scores_all):
                    if len(sc) == 1: sc = sc[0]
                    if len(sc) < len(label):
                        sc = np.pad(sc, (0, len(label)-len(sc)), mode='constant')
                    sc_all.append(sc)
                    # print(f'ind {i} has {len(sc)}, {sc.dtype}')  

                # print('chk smd norma here', np.array(sc_all).shape)
                if dim is None:
                    scores.append(np.mean(sc_all, axis=0))
                else:
                    scores.append(sc_all[dim])
                # clfs.append(clf_all[0])
                slabels.append(method)
        else:
            scores.append(scores_all[0])
            slabels.append(method)

        result_t = result_f1_acc(method_to, scores, label)
        result_t['machine1'] = machine1
        result_t['machine2'] = machine2
        results = pd.concat([results, result_t])

    
    return df_ID, label, scores, slabels , results, scores_all, clfs


def load_score_clf(dir, sel_data, method, conditions, province=None, sel_id=None, w_range=4, y_weight=0.5, w_weight=0.5, rollback=None, flip_chk=False):
    norma_clf = False
    print("Data->", sel_data, dir)
    filelists = os.listdir(f'{dir}{sel_data}/')
    print('Load score', method, len(filelists))
    
    weight_text = f'r_{w_range}_y_{y_weight}_w_{w_weight}'
    f_lists = [f for f in filelists if f.startswith(method)]
    if flip_chk:
        f_list_score = [f for f in f_lists if f.endswith('flip_anom_1__scores.pickle')]
    else:
        f_list_score = [f for f in f_lists if f.endswith('scores.pickle')]
    f_list_score.sort()
    # print(f_list_score)
    if 'AnDri' in method:
        f_list_off = [f for f in f_lists if f.endswith('off_.pickle')]
        f_list_off.sort()
        f_list_on = [f for f in f_lists if f.endswith('on_.pickle')]
        f_list_on.sort()

        if rollback is not None:
            if rollback:
                f_list_score = [f for f in f_list_score if '_rollback_False_' not in f]
                f_list_off = [f for f in f_list_off if '_rollback_False_' not in f]
                f_list_on = [f for f in f_list_on if '_rollback_False_' not in f]
            else:
                rollback_text = f'_rollback_{rollback}_'
                f_list_score = [f for f in f_list_score if rollback_text in f]
                f_list_off = [f for f in f_list_off if rollback_text in f]
                f_list_on = [f for f in f_list_on if rollback_text in f]

        # print('LL', len(filelists), len(f_list_off))

    elif method in ['NormA', 'SAND']:
        f_list_clf = [f for f in f_lists if f.endswith('clf_.pickle')]
        f_list_clf.sort()
        # print('NormA or SAND', f_list_clf)
        
        if len(f_list_clf) == 0:
            # print(f'No clf files?')
            norma_clf = True

    if sel_data == 'climate':
        stations_s = [f for f in f_list_score if f.split('_')[2]==province]
        sub_list_s = list(set([f for f in stations_s if f.split('_')[3] == str(sel_id)]))
        sub_list_s.sort()

        if 'AnDri' in method:
            print('Conditions:', conditions)
            print(province)
            stations_off = [f for f in f_list_off if f.split('_')[2]==province]
            sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id)]))
            sub_list_off.sort()

            stations_on = [f for f in f_list_on if f.split('_')[2]==province]
            sub_list_on = list(set([f for f in stations_on if f.split('_')[3] == str(sel_id)]))
            sub_list_on.sort()

            for sub_s, sub_off, sub_on in zip(sub_list_s, sub_list_off, sub_list_on):
                params = get_andri_param(sub_s)
                
                # print(conditions)
                try:
                    if (conditions['d'] == params['d'] and conditions['k'] == params['k'] and conditions['linkage'] == params['linkage'] and conditions['Wmax'] == params['Wmax'] and conditions['delta'] == params['delta'] and conditions['rmin'] == params['rmin']):
                        # print(sub_s)
                        clf = []
                        score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_off}'))
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_on}'))
                        print(params)
                        print(sub_s)
                        print(sub_off)
                        print(sub_on)
                        break
                except:
                    score, clf  = [], []
        elif method in ['NormA', 'SAND']:
            if norma_clf:
                for sub_s in sub_list_s:
                    try:
                        if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                            score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                            print(sub_s)
                                
                    except:
                        score, clf = [], []
                clf = []
            else:
                stations_off = [f for f in f_list_clf if f.split('_')[2]==province]
                sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id)]))
                print('clf off', sub_list_off)
                for sub_s, sub_off in zip(sub_list_s, sub_list_off):
                    try:
                        if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                            score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                            print(sub_s)

                            clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                            print(sub_off)
                            break
                    except:
                        score, clf = [], []
                if (len(clf) ==0) or (len(score) ==0):
                    score, clf = [], []
                
        else:
            for sub_s in sub_list_s:
                score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                # clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                clf = []
                break

    elif sel_data == 'traffic':
        # print('Score:', f_list_score[0].split('_')[2], type(province))
        if w_weight is None:
            stations_s = [f for f in f_list_score if int(f.split('_')[2])==province]     ## StatinoID for traffic
            sub_list_s = list(set([f for f in stations_s if f.split('_')[3] == sel_id.split('_')[0]]))
        else:
            stations_s = [f for f in f_list_score if (weight_text in f) and (int(f.split('_')[2])==province)]     ## StatinoID for traffic
            sub_list_s = list(set([f for f in stations_s if f.split('_')[3] == sel_id.split('_')[0]]))
        print(f'ID: {province}, Data: {sel_id}')
        sub_list_s.sort()

        if 'AnDri' in method:
            found = False
            print('Conditions:', conditions)
            if w_weight is None:
                stations_off = [f for f in f_list_off if int(f.split('_')[2])==province]
                sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id.split('_')[0])]))
                sub_list_off.sort()

                stations_on = [f for f in f_list_on if int(f.split('_')[2])==province]
                sub_list_on = list(set([f for f in stations_on if f.split('_')[3] == str(sel_id.split('_')[0])]))
                sub_list_on.sort()
            else:
                stations_off = [f for f in f_list_off if (weight_text in f) and int(f.split('_')[2])==province]
                sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id.split('_')[0])]))
                sub_list_off.sort()
                    
                
                stations_on = [f for f in f_list_on if (weight_text in f) and int(f.split('_')[2])==province]
                sub_list_on = list(set([f for f in stations_on if f.split('_')[3] == str(sel_id.split('_')[0])]))
                sub_list_on.sort()

            # print('L:', len(sub_list_s), len(sub_list_off), len(sub_list_on))

            for sub_s, sub_off, sub_on in zip(sub_list_s, sub_list_off, sub_list_on):
                params = get_andri_param(sub_s)
                print(params)
                # print(conditions)
                try:
                    if (conditions['d'] == params['d'] and conditions['k'] == params['k'] and conditions['linkage'] == params['linkage'] and conditions['Wmax'] == params['Wmax'] and conditions['delta'] == params['delta'] and conditions['rmin'] == params['rmin']):
                        # print(sub_s)
                        clf = []
                        score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_off}'))
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_on}'))
                        print(sub_s)
                        print(sub_off)
                        print(sub_on)
                        found=True
                        break
                except:
                    score, clf  = [], []

            if found is not True:
                score, clf = [], []

        elif method in ['NormA', 'SAND']:
            if norma_clf:
                for sub_s in sub_list_s:
                    try:
                        if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                            score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                            print(sub_s)

                    except:
                        score, clf = [], []
                clf = []
            else:
                if w_range is None:
                    stations_off = [f for f in f_list_clf if (weight_text in f) and int(f.split('_')[2])==province]
                    sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id.split('_')[0])]))
                else:
                    stations_off = [f for f in f_list_clf if (weight_text in f) and int(f.split('_')[2])==province]
                    sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id.split('_')[0])]))
                print('clf off', sub_list_off)
                for sub_s, sub_off in zip(sub_list_s, sub_list_off):
                    try:
                        if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                            score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                            print(sub_s)


                            clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                            print(sub_off)
                            break
                    except:
                        score, clf = [], []
                if (len(clf) ==0) or (len(score) ==0):
                    score, clf = [], []

        else:
            for sub_s in sub_list_s:
                score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                # clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                clf = []
                break

    return score, clf


def recompute_score_by_thresholds(method, save_dir, score_dir, save_to_dir, param, sel_param=None, sel_rev = False, traffic_choose='total_flow', run_m=False, w_range=4, y_weight=0.5, w_weight=0.5, chk_sel='gap', rollback=True, flip_chk=False):
    if not isinstance(rollback, bool):
        raise ValueError("rollback must be True or False")

    
    filelist_t = os.listdir(save_dir)
    print(save_dir, len(filelist_t))

    if 'precip' in save_dir:
        filelist = [f for f in filelist_t if f.endswith('processed.csv')]

        PROVINCES = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_map = {
            'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
            'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
        }
        d_name = 'climate'
    elif '101_N' in save_dir:
        filelist = [f for f in filelist_t if f.startswith('rev_')]
        df_stat = pd.read_csv('/home/parkj182/research/PeMS/stat_missing_station.csv', index_col=None)
        stationIDs = df_stat[(df_stat['all_len']>=149396) & (df_stat['missing_hour']<=103)]['stationID'].values
        d_name = f'traffic_{traffic_choose}'
        if len(filelist) < len(stationIDs):
            stationIDs = [int(f.split('_')[11]) for f in filelist]
        print('101_N: ', len(filelist), len(stationIDs))

    if method == 'AnDri':
        method_to = ['AnDri (off)', 'AnDri (on)']
        normalize = param['d']
        k=param['k']
        linkage = param['linkage']
        Wmax = param['Wmax']
        delta = param['delta']
        rmin= param['rmin']
    elif method in ['NormA', 'SAND']:
        method_to = [method]
        normalize = param['d']
    else:
        method_to = [method]

    results = pd.DataFrame()

    if 'precip' in save_dir:
        for sel_province in PROVINCES:
            sel_filelist = [f for f in filelist if f.split('_')[0] == sel_province]
            sel_IDs = list(set([f.split('_')[1] for f in sel_filelist]))
            # sel_IDs = list(selected_set[selected_set['province']==sel_province]['stationID'])
            sel_IDs.sort()
            print(sel_IDs)

            if method == 'AnDri':
                conditions = {'d': normalize, 'k':k, 'linkage':linkage,'Wmax':Wmax, 'delta':delta, 'rmin':rmin}
            elif method in ['NormA', 'SAND']:
                conditions = {'d': param['d']}
            else:
                conditions = {}

            for sel_id_idx in range(len(sel_IDs)):
                print(f'Province: {sel_province}-> {sel_IDs[sel_id_idx]}, Stations: {sel_id_idx+1}/{len(sel_IDs)}')

                ## load scores and models
                # try:
                score, clf = load_score_clf(score_dir, 'climate', method, conditions, sel_province, sel_IDs[sel_id_idx], w_range, y_weight, w_weight, rollback if method == 'AnDri' else None, flip_chk)
                if sel_rev and method == 'AnDri':
                    score = []
                    score.append(clf[0].scores_rev)
                    score.append(clf[1].scores_rev)

                if run_m and method == 'AnDri':
                    score_t = score.copy()
                    score_r = []
                    for s in score_t:
                        s2 = running_mean(s, 24)
                        s2 = s2[:len(s)]
                        s3 = [s2[0]]*12 + list(s2) + [s2[-1]]*11
                        score_r.append(np.array(s3))
                    score = score_r


                ## Data
                f_list = [f  for f in filelist if f.split('_')[1] == str(sel_IDs[sel_id_idx])]
                print(f_list, len(score))
                if len(f_list) ==0: continue
                df_ID = pd.read_csv(f'{save_dir}/{f_list[0]}')
                df_ID['datetime'] = pd.to_datetime(df_ID['Date/Time (LST)'], format='%Y-%m-%d %H:%M')
                df_ID = df_ID[df_ID['datetime']>=datetime.datetime(2023, 7, 1)]

                sel_station = df_ID['Station Name'].iloc[0]
                # print(sel_province, sel_IDs[sel_id_idx], sel_station)

                ### compute accuracy with revised labels
                # display(df_ID.head())
                # result_t = result_f1_acc(method_to, score, df_ID[f'heavy_{5}'].to_numpy())
                # result_t['province'] = sel_province
                # result_t['station'] = sel_station
                # result_t['stationID'] = sel_IDs[sel_id_idx]
                # result_t['threshold'] = f'TH_{5}'
                # results = pd.concat([results, result_t])
                for i in range(6):
                    try:
                        result_t = result_f1_acc(method_to, score, df_ID[f'heavy_{i}'].to_numpy())
                        result_t['province'] = sel_province
                        result_t['station'] = sel_station
                        result_t['stationID'] = sel_IDs[sel_id_idx]
                        result_t['threshold'] = f'TH_{i}'
                        results = pd.concat([results, result_t])
                    except:
                        continue
        
    elif '101_N' in save_dir:
        if method == 'AnDri':
            conditions = {'d': normalize, 'k':k, 'linkage':linkage,'Wmax':Wmax, 'delta':delta, 'rmin':rmin}
        elif method in ['NormA', 'SAND']:
            conditions = {'d': param['d']}
        else:
            conditions = {}
        for id in stationIDs:
            print(f'Station: {id}, choose: {traffic_choose}')
            score, clf = load_score_clf(score_dir, 'traffic', method, conditions, id, traffic_choose, w_range, y_weight, w_weight, rollback if method == 'AnDri' else None)
            if sel_rev and method == 'AnDri':
                score = []
                score.append(clf[0].scores_rev)
                score.append(clf[1].scores_rev)

            if run_m and method == 'AnDri':
                score_t = score.copy()
                score_r = []
                for s in score_t:
                    s2 = running_mean(s, 24)
                    s2 = s2[:len(s)]
                    s3 = [s2[0]]*12 + list(s2) + [s2[-1]]*11
                    score_r.append(np.array(s3))
                score = score_r

            ## Data
            if chk_sel == 'gap':
                id_pos = 11
                f_list = [f for f in filelist if f.split('_')[id_pos] == str(id)]
            elif chk_sel == None:
                id_pos = 3
                f_list = [f for f in filelist if f.split('_')[id_pos] == str(id)]
            elif chk_sel == 'weight':
                id_pos = 9
                f_list = [f for f in filelist if f.split('_')[id_pos] == str(id)]
            
            print(f_list, len(score))
            if len(f_list) == 0 or len(score) == 0: continue
            df_ID = pd.read_csv(f'{save_dir}/{f_list[0]}')
            df_ID['timestamp'] = pd.to_datetime(df_ID['timestamp'])
            df_ID = df_ID[df_ID['timestamp'] >= datetime.datetime(2019, 1, 1)]
            label = df_ID[f'{traffic_choose}_label'].to_numpy()
            result_t = result_f1_acc(method_to, score, label)
            result_t['stationID'] = id
            results = pd.concat([results, result_t])


    if sel_rev and method == 'AnDri':
        if sel_param is not None:
            results.to_csv(f'{save_to_dir}/revised_thresholds_{d_name}_{method}_{sel_param}_r_{run_m}_k_{k}_rmin_{rmin}_linkage_{linkage}_rollback_{rollback}rev.csv', index=None)
        else:
            results.to_csv(f'{save_to_dir}/revised_thresholds_{d_name}_{method}_r_{run_m}_k_{k}_wr_{w_range}_rmin_{rmin}_linkage_{linkage}_rollback_{rollback}.csv', index=None)    
    else:
        if method == 'AnDri':
            results.to_csv(f'{save_to_dir}/revised_thresholds_{d_name}_{method}_r_{run_m}_k_{k}_wr_{w_range}_rmin_{rmin}_linkage_{linkage}_rollback_{rollback}.csv', index=None)
        else:
            results.to_csv(f'{save_to_dir}/revised_thresholds_{d_name}_{method}_wr_{w_range}.csv', index=None)
    return results


def draw_selected_station(save_dir, score_dir, sel_province, stationID, threshold=5, methods = 'all', k=1, linkage='ward', rmin=0.02, rollback=None):
    filelist_t = os.listdir(save_dir)
    filelist = [f for f in filelist_t if f.endswith('processed.csv')]

    if methods == 'all':
        methods = ['AnDri', 'CABD', 'NormA', 'SAND', 'TranAD', 'ARCUS', 'DIVAD']
    else:
        methods = methods


    scores, slabels = [], []
    clfs = []
    results = pd.DataFrame()
    for method in methods:
        if method == 'AnDri':
            method_to = ['AnDri (off)', 'AnDri (on)']
            normalize = 'zero-mean'
            # k=3
            # linkage = 'ward'
            Wmax = 20
            delta = 10
            # rmin= 0.02
        elif method in ['NormA', 'SAND']:
            normalize = 'zero-mean'
            method_to = [method]
        else:
            method_to = [method]

        sel_filelist = [f for f in filelist if f.split('_')[0] == sel_province]
        sel_IDs = list(set([f.split('_')[1] for f in sel_filelist]))
        # sel_IDs = list(selected_set[selected_set['province']==sel_province]['stationID'])
        sel_IDs.sort()
        print(sel_IDs)

        if method == 'AnDri':
            if rollback is None:
                conditions = {'d': normalize, 'k':k, 'linkage':linkage,'Wmax':Wmax, 'delta':delta, 'rmin':rmin}
            else:
                conditions = {'d': normalize, 'k':k, 'linkage':linkage,'Wmax':Wmax, 'delta':delta, 'rmin':rmin, 'rollback':True}
        elif method in ['NormA', 'SAND']:
            conditions = {'d': normalize}
        else:
            conditions = {}

        print(f'Province: {sel_province}-> {stationID}, Stations: {stationID}')

        ## load scores and models
        if rollback is not None:
            score, clf = load_score_clf(score_dir, 'climate', method, conditions, sel_province, stationID, rollback=rollback)
        else:
            score, clf = load_score_clf(score_dir, 'climate', method, conditions, sel_province, stationID)
        if method == 'AnDri':
            scores.append(score[0])
            scores.append(score[1])
            clfs.append(clf[0])
            clfs.append(clf[1])
            slabels.append('AnDri (off)')
            slabels.append('AnDri (on)')
        else:
            # print(score.shape)
            scores.append(score)
            clfs.append(clf)
            slabels.append(method)

        print('chk:', len(score[0]))
        ## Data
        f_list = [f  for f in sel_filelist if f.split('_')[1] == str(stationID)]


        df_ID = pd.read_csv(f'{save_dir}/{f_list[0]}')
        df_ID['datetime'] = pd.to_datetime(df_ID['Date/Time (LST)'], format='%Y-%m-%d %H:%M')
        df_ID = df_ID[df_ID['datetime']>=datetime.datetime(2023, 7, 1)]
        # display(df_ID.head())
        data = df_ID['Precip. Amount (mm)'].to_numpy()
        label = df_ID[f'heavy_{threshold}'].to_numpy()
        sel_station = df_ID['Station Name'].iloc[0]

        ### compute accuracy with revised labels
        # for i in range(5):
        result_t = result_f1_acc(method_to, np.array(score), label)
        result_t['province'] = sel_province
        result_t['station'] = sel_station
        result_t['stationID'] = stationID
        result_t['threshold'] =f'TH_{threshold}'
        results = pd.concat([results, result_t])


    
    return data, label, scores, slabels, clfs, results, df_ID
