import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import argparse

from util.util_andri import find_length 
from util.util_exp import result_f1_acc, save_pickle, get_data, get_multi_data

# import torch
import time

from util.TSB_AD.models.andri import AnDri

## norma.py: Ask to the original paper author's to test the NORMA
from util.TSB_AD.models.norma import NORMA

## Original file of sand.py from https://github.com/TheDatumOrg/TSB-UAD
from util.TSB_AD.models.sand import SAND

## Original file of error_detection.py from https://github.com/szamani20/time-series
from util.TSB_AD.models.error_detection import ErrorDetection

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-data', help='  : select datasets (ex. all, INN_Sensor, SMD, ...)', default='all')
parser.add_argument('-method', help='      : select test methods (ex. all, TranAD, AnDri, ...)', default='all')

## Parameters for AnDri, NormA, and SAND
parser.add_argument('-nm_len', help='      : length of normal pattern (ex. 2)', default=2)
parser.add_argument('-normalize', help='      : normalization method (z-norm, zero-mean, euclidean, z-norm_rev, dtw, sbd)', default='zero-mean')
parser.add_argument('-dist_org', help='      : use original distance in NormA (True/False)', default=True)

## Parameters for SAND only
parser.add_argument('-start_chunk', help='      : SAND start_chunk', default=5000)
parser.add_argument('-chunk_size', help='      : SAND chunk_size', default=2000)

## Parameters for AnDri only
parser.add_argument('-k', help='      : AnDri parameter k', default=1)
parser.add_argument('-linkage', help='      : AnDri linkage method (single, complete, average, ward)', default='ward')
parser.add_argument('-max_W', help='      : AnDri parameter Wmax', default=20)
parser.add_argument('-delta_max', help='      : AnDri parameter delta_max, for WitnessPair', default=10)
parser.add_argument('-rmin', help='      : AnDri parameter min_size of appropriate cluster', default=0.02)
parser.add_argument('-step', help='         : Stepwise anomaly score computation', default=True)

## Parameters for TranAD
# parser.add_argument('-epochs', help='       : TranAD or DNNs epochs', default=10)
# parser.add_argument('-smd_dim', help='      : SMD slice dimension', default = 1)

## For evalutation purpose only
parser.add_argument('-climate_short', default=True)         # for climate
parser.add_argument('-traffic_choose', default='total_flow')    ## for traffic
parser.add_argument('-w_range', default='4')    ## for traffic
parser.add_argument('-y_weight', default='0.5')    ## for traffic
parser.add_argument('-w_weight', default='0.5')    ## for traffic
parser.add_argument('-rollback', default=True)
parser.add_argument('-norma_pe', default=1)
# parser.add_argument('-overlap', help='      : Proportion of overlapping (0: no overlap, 0.5: half). Should be less than 1', default=0)

args = parser.parse_args()


def args_passing(args):
    ## args
    datasets = test_datasets if args.data == 'all' else [args.data]
    test_methods = selected_methods if args.method == 'all' else [args.method]

    nm_len = int(args.nm_len)
    kadj = int(args.k)
    normalize = args.normalize
    linkage = args.linkage
    max_W = int(args.max_W)
    delta_max = int(args.delta_max)
    rmin = float(args.rmin)
    stepwise = True if args.step in ['True', 'true', 'T', True] else False
    if stepwise:
        print('STEPWISE: ', stepwise)
    # tranad_epochs = int(args.epochs)
    dist_org = args.dist_org in ['True', 'true', '1', 'T', 't']
    start_chunk = int(args.start_chunk)
    chunk_size = int(args.chunk_size)
    
    short = True if args.climate_short in ['True', 'true', 'T', True] else False
    rollback = True if args.rollback in ['True', 'true', 'T', True] else False
    norma_pe = float(args.norma_pe)

    traffic_choose = args.traffic_choose
    w_range = int(args.w_range)
    y_weight = float(args.y_weight)
    w_weight = float(args.w_weight)

    return (
        datasets, 
        test_methods, 
        nm_len, 
        kadj,
        normalize, 
        linkage, 
        max_W, 
        delta_max, 
        rmin, 
        stepwise, 
        dist_org, 
        start_chunk, 
        chunk_size, 
        short,
        traffic_choose,
        w_range,
        y_weight,
        w_weight,
        rollback,
        norma_pe,
    )


####################################################################################
## Global

MAX_SCORE = 1000
test_datasets = ['climate', 'INN_Sensor', 'SMD', 'traffic', 'SWaT']
selected_methods= ['TranAD', 'ARCUS', 'INN', 'NormA', 'SAND', 'DAMP', 'AnDri']


save_dir = './results/'      ## To be identified

overlap = 0
stepwise=True


def main(argv, args):
    print('\n')
    print('argv: ', argv)
    print('args: ', args)

    ## args
    (
        datasets,
        test_methods,
        nm_len,
        kadj,
        normalize,
        linkage,
        max_W,
        delta_max,
        rmin,
        stepwise,
        dist_org,
        start_chunk,
        chunk_size,
        short,
        traffic_choose,
        w_range,
        y_weight,
        w_weight,
        rollback,
        norma_pe,
    ) = args_passing(args)

    ## Test datasets
    for data_name in datasets:
        if data_name == 'climate':
            data_list, label_list, filelist, provinces, stations = get_data(data_name, short=short)
        elif data_name == 'traffic':
            data_list, label_list, filelist, filelist_th, stationIDs = get_data(
                data_name, sel_columns=1, traffic_sel=traffic_choose, 
                w_range=w_range, y_weight=y_weight, w_weight=w_weight)
        elif data_name in ['SMD', 'SWaT', 'WADI']:
            data_list, label_list, filelist = get_multi_data(data_name)
        else:
            data_list, label_list, filelist = get_data(data_name)

        print(f'{data_name} has {len(data_list)} files')
        if test_methods == ['AnDri']:
            time_all = pd.DataFrame(
                columns=['filename','Offline_time', 'Online_time', 'Offline_flip', 'Offline_cnt', 
                         'Offline_stat', 'Online_flip'])
        else:
            time_all = pd.DataFrame(
                columns=['filename','Offline_time', 'Online_time', 'Offline_flip', 'Online_flip'])
        idx = 0
        for data, label, file_name in zip(data_list, label_list, filelist):
            scores_all, labels_all = [], []
            data = np.nan_to_num(data, nan=0)
            
            ## Specific periodic length cases
            if data_name in ['Weather', 'climate', 'traffic']:
                slidingWindow = 24
            ## It needs short l
            elif data_name in ['INN_Sensor']:
                slidingWindow = 10
            ## Limit the range between 48 and 300
            else:
                slidingWindow = find_length(data)
                while slidingWindow < 48:
                    slidingWindow = slidingWindow*2
                if slidingWindow > 300:
                    slidingWindow = 300

            print('SlidingWindow:', slidingWindow)
            # delta = max_W*slidingWindow

            ## Set train_len as 20% in general
            train_len = int(len(data) * 0.2)
            ## For traffic data exp., we limit train len as 1-year (before COVID-era)
            if data_name == 'traffic':
                train_len = 24*365
            scores_rev = []

            print(file_name)
            if test_methods == ['AnDri']:
                
                data, label = data.reshape(-1), label.reshape(-1)

                ## Offline
                start_t = time.time()
                clf_off = AnDri(pattern_length=slidingWindow, normalize=normalize, linkage_method = linkage, 
                                th_reverse=5, kadj=kadj, nm_len=nm_len, overlap=0, max_W=max_W, delta_max=delta_max, eta=1)
                clf_off.fit(data, y=label, online=False, training_len=int(train_len), stepwise=stepwise, min_size=rmin, rollback=rollback)
                end_t = time.time()
                offline_time = end_t - start_t
                print(f'Offline time: {offline_time:.1f} sec')

                ### Cannot get any validate normal patterns
                if len(clf_off.scores) == 0:
                    # num_min_cl_off = 0
                    ValueError(f'No normal pattern detected: change the parameters, rmin:{rmin}')
                    scores_all.append(np.zeros(len(data)))
                else:
                    num_min_cl_off = len(clf_off.listcluster[clf_off.listcluster == -1])
                    print(f'R_size (off): {num_min_cl_off}')
                    score = clf_off.scores_rev
                    score = np.nan_to_num(score, nan=np.nanmax(score))
                    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
                    if len(score) < len(data):
                        score = np.append(score, np.ones(len(data) -len(score))*np.mean(score))
                    scores_all.append(score[:len(data)])
                labels_all.append(label)

                ## For analyzing AHC
                off_flip = clf_off.num_flip
                off_cnt = clf_off.count_flip
                off_stat = clf_off.flips
                print(f'Number of rollbacks (off): {off_flip}')


                ## Online
                start_t = time.time()
                clf_on = AnDri(pattern_length=slidingWindow, normalize=normalize, linkage_method = linkage, 
                                th_reverse=5, kadj=kadj, nm_len=nm_len, overlap=0, max_W=max_W, delta_max=delta_max, eta=1)
                clf_on.fit(data, y=label, online=True, training_len=int(train_len), stepwise=stepwise, min_size=rmin, rollback=rollback)
                end_t = time.time()
                online_time = end_t - start_t
                if len(clf_on.scores) == 0:
                    num_min_cl_on = 0
                    print(f'R_size (on): {num_min_cl_on}')
                    scores_all.append(np.zeros(len(data)))
                else:    
                    num_min_cl_on = len(clf_on.listcluster[clf_on.listcluster == -1])
                    print(f'R_size (on): {num_min_cl_on}')
                    score_on = clf_on.scores_rev
                    # score_on[np.isnan(score_on)] = np.nanmax(score_on)
                    score_on = np.nan_to_num(score_on, nan=np.nanmax(score_on))
                    score_on = MinMaxScaler(feature_range=(0,1)).fit_transform(score_on.reshape(-1,1)).ravel()
                    if len(score_on) < len(data):
                        score_on = np.append(score_on, np.ones(len(data)-len(score_on))*np.mean(score_on))
                    scores_all.append(score_on[:len(data)])
                labels_all.append(label)
                on_flip = clf_on.num_flip
                print(f'Number of flip merges (on): {on_flip}')
                time_all.loc[idx] = [file_name, offline_time, online_time, off_flip, off_cnt, off_stat, on_flip]
                idx +=1
                # torch.cuda.empty_cache()
                # except:
                    # continue

            elif test_methods == ['NormA']:
                print('Test: NORMA-OFF')
                
                data = data.reshape(-1)
                label = label.reshape(-1)
                normalize_comp = 'z-norm' if dist_org else normalize

                start_t = time.time()
                clf = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow, percentage_sel=norma_pe, normalize=normalize_comp)
                clf.fit(data)
                end_t = time.time()
                process_time = end_t - start_t
                
                print('NormA-Done (takes)', end_t - start_t)
                score = clf.decision_scores_
                score = np.nan_to_num(score, nan=np.nanmax(score))
                score = np.nan_to_num(score, posinf=MAX_SCORE, neginf=0)
                if score is None or len(score) == 0:
                    print('NormA Error: score is None')
                    scores_all.append(np.zeros(len(data)))
                    continue
                score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                scores_all.append(score[:len(data)])
                labels_all.append(label[:len(data)])
                time_all.loc[idx] = [file_name, process_time, process_time, 0, 0]
                idx +=1
                # torch.cuda.empty_cache()

            elif test_methods == ['SAND']:
                print('Test: SAND-Online')
                data = data.reshape(-1)
                label = label.reshape(-1)

                if slidingWindow > 250: 
                    if data_name not in ['WADI', 'SWaT']:
                        start_chunk = start_chunk*2
                        chunk_size = chunk_size*2

                start_t = time.time()
                clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow), normalize=normalize)
                x = data
                try:
                    clf.fit(x,online=True,alpha=0.5,init_length=(start_chunk // slidingWindow +1)*slidingWindow,batch_size=(chunk_size//slidingWindow +1)*slidingWindow,verbose=True, overlaping_rate=int(4*slidingWindow))
                    end_t = time.time()
                    process_time = end_t - start_t
                    print('SAND-Done (takes)', end_t - start_t)
                    score = clf.decision_scores_
                    score = np.nan_to_num(score, nan=np.nanmax(score))
                    score = np.nan_to_num(score, posinf=MAX_SCORE, neginf=0)
                    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
                    # score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
                    scores_all.append(score[:len(data)])
                    labels_all.append(label[:len(data)])
                    time_all.loc[idx] = [file_name, process_time, process_time, 0, 0]
                    idx +=1
                except:
                    scores_all.append(np.zeros(len(data)))
                    labels_all.append(label[:len(data)])
                    time_all.loc[idx] = [file_name, -1, -1, 0, 0]
                    idx +=1
                    continue

            elif test_methods == ['CABD']:
                print('Test: INN-OFF')
                data = data.reshape(-1)
                label = label.reshape(-1)
                start_t = time.time()
                indices = np.arange(len(data))
                ts = np.column_stack((indices, data, label))
                ed = ErrorDetection(ts)
                score = ed.compute_anomaly_score(ts[:,1])
                mad = ed.compute_median_absolute_deviation(score)
                end_t = time.time()
                process_time = end_t - start_t
                time_all.loc[idx] = [file_name, process_time, process_time, 0, 0]
                idx +=1
                print('INN-Done (takes)', end_t - start_t)
                candidate_points = [i for i in range(2, len(ts)) if ed.compute_median_absolute_deviation(ts[i-2:i+1,1]) > mad]
                scores_all.append(score[:len(data)])
                labels_all.append(label[:len(data)])
                

            for s in scores_all:
                # print(len(s), len(label))
                if np.isnan(s).any():
                    s = np.nan_to_num(s, nan=0.0)
                scores_rev.append(s)
            

            if data_name == 'INN_Sensor':
                pre_name = test_methods[0] + '_' + data_name + file_name.split(' ')[-1] 
            elif data_name == 'climate':
                pre_name = test_methods[0] + '_' + data_name + '_' + provinces[filelist.index(file_name)] + '_' + stations[filelist.index(file_name)]
                # pre_name = data_name + '_' + provinces[filelist.index(file_name)] + '_' + stations[filelist.index(file_name)] + '_anom_1_'
            elif data_name == 'ASD':
                pre_name = test_methods[0] + '_' + file_name.split('.')[0] 
            else:
                pre_name = test_methods[0] + '_' + file_name.split('.')[0] 

            if test_methods == ['AnDri']:
                pre_name = (
                    pre_name + '_l_' + str(slidingWindow) + '_d_' + str(normalize) 
                    + '_nm_' + str(nm_len) + '_k_' + str(kadj) + '_linkage_' + str(linkage) + '_Wmax_' + str(max_W) 
                    + '_delta_' + str(delta_max) + '_rmin_' + str(rmin) + '_rollback_' + str(rollback) 
                    + '_step_' + str(stepwise) + 'flip_anom_1_'
                )
                save_name = pre_name + '_off_' 
                save_pickle(f'{save_dir}{data_name}/{save_name}.pickle', clf_off)
                save_name = pre_name + '_on_'  
                save_pickle(f'{save_dir}{data_name}/{save_name}.pickle', clf_on)
            elif test_methods == ['NormA'] or test_methods == ['SAND']:
                pre_name = pre_name + '_l_' + str(slidingWindow) + '_d_' + str(normalize) + '_nm_' + str(nm_len)  + '_pe_'+ str(norma_pe) + '_anom_1_'
                save_name = pre_name + '_clf_' 
                save_pickle(f'{save_dir}{data_name}/{save_name}.pickle', clf)
            else:
                pre_name = pre_name + '_l_' + str(slidingWindow) + '_d_' + str(normalize) + '_nm_' + str(nm_len) + '_anom_1_'

            save_name = pre_name + '_scores' 
            save_pickle(f'{save_dir}{data_name}/{save_name}.pickle', scores_rev)

            print('Save:', save_name)

            save_name = pre_name + '_labels'
            save_pickle(f'{save_dir}{data_name}/{save_name}.pickle', labels_all)

            if test_methods == ['AnDri']:
                save_methods = ['AnDri (off)', 'AnDri (on)']
            else:
                save_methods = test_methods
            result_org = result_f1_acc(save_methods, scores_rev, label)
            result_org['file'] = file_name.split('.')[0]
            save_name = pre_name + '_result'
            result_org.to_csv(f'{save_dir}{data_name}/{save_name}.csv')
        
        time_name = pre_name + '_test_time'
        time_all.to_csv(f'{save_dir}{data_name}/{time_name}_flip.csv')


if __name__ == '__main__':
    argv = sys.argv
    main(argv, args)
        

