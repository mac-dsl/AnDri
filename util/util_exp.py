import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.TSB_AD.metrics import metricor
from sklearn import metrics
# from util.TSB_AD.slidingWindows import find_length #,plotFig, printResult
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import os
import sys
import copy

from util.plot_aadd import plotFigRev, find_anomaly_intervals
from util.util_a2d2 import find_length, divide_subseq, norm_seq, running_mean
# from util.ahc import adaptive_ahc2

# from scipy.io import arff
import arff
import warnings
import random

# from tqdm.notebook import tqdm
import time
import math
from util.TranAD_base import *
from util.TSB_AD.models.norma import NORMA
from util.TSB_AD.models.a2d2 import A2D2
from util.TSB_AD.models.sand import SAND
from util.TSB_AD.models.damp import DAMP
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray', 
          'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan', 'pink', 'deepskyblue', 'lawngreen',
          'royalblue', 'darkgrey', 'darkorange', 'darkgreen','darkviolet','salmon','olivedrab','lightcoral','darkcyan','yellowgreen']
markers = ['o', 'x', '^', 'v', 's', '*', '+', '.', ',', '<', '>' , '1','2','3','4','p','h','H','D','d']
warnings.filterwarnings('ignore')

rst_columns =['AUC', 'R_AUC', 'Precision', 'Recall', 'F1', 'AP', 'R_AP', 'RPrecision', 'RRecall']


def read_arff(filename):
    """
    Find ndarray corresponding to data and labels from arff data
    """
    arff_content = arff.load(f.replace(',\n', '\n') for f in open(filename, 'r'))
    arff_data = arff_content['data']
    data = np.array([i[:1] for i in arff_data])
    anomaly_labels = np.array([i[-1] for i in arff_data])
    anomaly_labels = anomaly_labels.reshape((len(anomaly_labels),1))
    return data.astype(float), anomaly_labels.astype(float)
    
def get_acc(label, score, slidingWindow, ths=None):
    result = pd.DataFrame(columns=rst_columns)
    grader = metricor()
    if np.sum(label) != 0:
        R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True)
        L = grader.metric_new(label, score, ths=ths)
        # print(L)
        
        precision, recall, AP = grader.metric_PR(label, score)
        # print(f'Precision? {precision}, Recall? {recall}'
        
        result.loc[0] = [L[0], R_AUC, L[1], L[2], L[3], AP, R_AP, L[4], L[7]]
        return result


def training_model(train_loader, test_loader, NN_model, labels, dataset):
    ########################################################################
    ## For training DNNs (in terms of TranAD and related references)

    model, optimizer, scheduler, epoch, accuracy_list = load_model(NN_model, labels.shape[1])
    
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
        
    ### Training phase
    print(f'{color.HEADER}Training {NN_model} on {dataset}{color.ENDC}')
    num_epochs = 5; e = epoch + 1; start = time()
    for e in list(range(epoch+1, epoch+num_epochs+1)):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
    save_model(model, optimizer, scheduler, e, accuracy_list)
	# plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
	# plot_accuracies(accuracy_list)
    return model, testD, testO, optimizer, scheduler

##########################################################################################
## Inject anomalies

def get_random_index(num, n_range, w, dist_anomaly = 'uniform', max_retry=10000):
    list_k = []
    i=0
    ret = 0
    if num <1: 
        print('Too Small Size', num)
        return None
    while(len(list_k) <=num):
        if dist_anomaly == 'uniform':
            k = random.randint(0, int(n_range/w))
        elif dist_anomaly == 'Gaussian':
            k = int(random.gauss(0, 0.5)*int(n_range/2/w)+int(n_range/2/w))

        elif dist_anomaly == 'rayleigh':
            k = int(np.random.rayleigh(scale = 2, size=1)*int(n_range/w)/10)
            # rayleigh dist. returns a random number 0 -10
        elif dist_anomaly == 'inv_rayleigh':
            k = int((10-np.random.rayleigh(scale = 2, size=1))*int(n_range/w)/10)
        
        if k <0: k = 0
        if k > int(n_range/w): k = int(n_range/w)-1
        if k not in list_k:
            list_k.append(k)
            i += 1
            if len(list_k) >=num: 
                break
        else:
            ret +=1
            if ret >= max_retry:
                print(f'OverCounted! {len(list_k)}/{num} // {int(n_range/w)}')
                return list_k
    return np.array(list_k)*w

## Inject anomalies with type (flat and/or scale), distribution of scale, and position (distribution)
def add_anomaly(data, label, p_flat, len_flat, p_scale, r_scale, len_scale, scale_dist='uniform', dist_anomaly='uniform'):
    data2, label2= copy.deepcopy(data), copy.deepcopy(label)
    len_data = len(data)
    if p_flat >0:
        # ind_flat = [random.randint(0, len_data) for i in range(int(p_flat*len_data/len_flat))]
        ind_flat = get_random_index(int(p_flat*len_data/len_flat), len_data, len_flat, dist_anomaly=dist_anomaly)
        for i in ind_flat:
            if i+len_flat < len(data):
                data2[i:i+len_flat] = data[i]
                label2[i:i+len_flat] = 1
            else:
                data2[i:] = data[i]
                label2[i:] = 1
    if p_scale >0:
        # ind_scale = [random.randint(0, len_data) for i in range(int(p_scale*len_data/len_scale))]
        ind_scale = get_random_index(int(p_scale*len_data/len_scale), len_data, len_scale, dist_anomaly=dist_anomaly)
        for i in ind_scale:
            if scale_dist == 'uniform':
                r_scale_t = random.uniform(r_scale/2, r_scale)
            elif scale_dist == 'Gaussian':
                r_scale_t = random.gauss(r_scale, 0.1)
            if i + len_scale < len(data):
                t_data = data[i:i+len_scale]
                m_data = np.mean(t_data)
                t_data = (t_data - m_data)*r_scale_t
                t_data += m_data
                data2[i:i+len_scale] = t_data
                label2[i:i+len_scale] = 1
            else:
                t_data = data[i:]
                m_data = np.mean(t_data)
                t_data = (t_data - m_data)*r_scale_t
                t_data += m_data
                data2[i:] = t_data
                label2[i:] = 1
                

    # plot_anomaly(data2, label2)
    return data2, label2

## Inject scale-anomalies. Put 'seq_a' x scale into selected positions (temp. for weather only)
def add_local_anomaly(data, label, seq_a, p_scale, r_scale, len_scale, scale_dist='uniform', dist_anomaly='uniform'):
    data2, label2= copy.deepcopy(data), copy.deepcopy(label)
    len_data = len(data)
    ind_scale = get_random_index(int(p_scale*len_data/len_scale), len_data, len_scale, dist_anomaly=dist_anomaly)
    for i in ind_scale:
        if scale_dist == 'uniform':
            r_scale_t = random.uniform(r_scale/2, r_scale)
        elif scale_dist == 'Gaussian':
            r_scale_t = random.gauss(r_scale, 0.1)
        # print(r_scale_t)
            
        if i + len_scale < len(data):
            t_data = seq_a[:48]
            # plt.plot(t_data)
            m_data = np.mean(t_data)
            t_data = (t_data - m_data)*r_scale_t
            t_data += np.mean(data2[i:i+len_scale])
            # print(f'{m_data} --> {np.mean(data2[i:i+len_scale])}')
            # plt.plot(t_data)
            data2[i:i+len_scale] = t_data
            label2[i:i+len_scale] = 1
        else:
            t_data = seq_a[i:]
            m_data = np.mean(t_data)
            t_data = (t_data - m_data)*r_scale_t
            # t_data += m_data
            t_data += np.mean(data2[i:])
            data2[i:] = t_data
            label2[i:] = 1
            
    # plot_anomaly(data2, label2)
    return data2, label2

## keep_number. True: maintain the number of anomalies (ex. 10% from all into data2) vs. False: maintain % into data2
def add_local_anomaly2(data1, data2, p_d1, p_scale, r_scale, len_scale, scale_dist='uniform', dist_anomaly='uniform', keep_number=False):
    data1, data2 = data1.reshape(-1), data2.reshape(-1)
    max_len = max(len(data1), len(data2))
    if min(len(data1), len(data2)) < max_len*(1-p_d1):
        print(f'Error! Data #{np.argmin([len(data1), len(data2)])+1} is less than {(1-p_d1)*100}%')
        return None
    
    data = np.append(data1[:int(max_len*p_d1)], data2[-int(max_len*(1-p_d1)):])
    len_d1, len_d2 = int(max_len*p_d1), int(max_len*(1-p_d1))
    len_data = len(data)
    print(f'Dist: {len_d1} + {len_d2} => {len_data}')
    data_t, label_t= copy.deepcopy(data), np.zeros(len(data))
    
    num_anomalies = int(p_scale*len_data/len_scale) if keep_number else int(p_scale*len_d2/len_scale)
        
    ind_scale = get_random_index(num_anomalies, len_d2, len_scale, dist_anomaly=dist_anomaly)
    # print('scale: ', ind_scale)
    count_t = 0
    for i in ind_scale:
        if scale_dist == 'uniform':
            r_scale_t = random.uniform(r_scale/2, r_scale)
        elif scale_dist == 'Gaussian':
            r_scale_t = random.gauss(r_scale, 0.1)
        # print(r_scale_t)
            
        if i + len_scale < len_d2:
            t_data = data1[i:i+len_scale]   ## Get seq. from data1
            m_data = np.mean(t_data)
            t_data = (t_data - m_data)*r_scale_t
            t_data += np.mean(data[i+len_d1:i+len_d1+len_scale]) ## position to inject
            # print(f'{m_data} --> {np.mean(data2[i:i+len_scale])}')
            # plt.plot(t_data)
            data_t[i+len_d1:i+len_d1+len_scale] = t_data
            label_t[i+len_d1:i+len_d1+len_scale] = 1
        else:
            count_t +=1
            print(f'Invalid position: {count_t}')
            
    # plot_anomaly(data2, label2)
    return data_t, label_t

def add_local_anomaly_no_guard(data1, data2, d1_len, d2_len, win1, win2, p_scale, r_scale, scale_dist='uniform', dist_anomaly='uniform', keep_number=False):
    data1, data2 = data1.reshape(-1), data2.reshape(-1)

    if d1_len > d2_len:
        ## put anomalies (similar to d2) into d1
        d_long, d_short = data1, data2
        len_l, len_s = d1_len, d2_len
        len_scale = win2
    else:
        d_long, d_short = data2, data1
        len_l, len_s = d2_len, d1_len
        len_scale = win1

    data = np.array([])
    d_inds = np.array([])
    i =0

    while i < len(data1) and i< len(data2):
        data = np.append(data, d_short[i:i+len_s])
        d_inds = np.append(d_inds, np.ones(len(d_short[i:i+len_s])))
        i = i+len_s
        data = np.append(data, d_long[i:i+len_l])
        d_inds = np.append(d_inds, np.ones(len(d_long[i:i+len_l]))*2)
        i = i+len_l

    add_noise_range = len(d_inds[d_inds==2])

    num_anomalies = int(p_scale*len(data)/len_scale) if keep_number else int(p_scale*add_noise_range/len_scale)
    print(f'Num. Anomalies: {num_anomalies}/{add_noise_range/len_scale} -- {len(data)}')
    ind_scale = get_random_index(num_anomalies, add_noise_range, len_scale, dist_anomaly=dist_anomaly)
    small_intervals = find_anomaly_intervals(abs(d_inds-2))

    count_t = 0
    data_t, label_t = copy.deepcopy(data), np.zeros(len(data))

    for i in ind_scale:
        if scale_dist == 'uniform':
            r_scale_t = random.uniform(r_scale/2, r_scale)
        elif scale_dist == 'Gaussian':
            r_scale_t = random.gauss(r_scale, 0.1)

        ## Find position to inject
        pos_i = i
        for (s_start, s_end) in small_intervals:
            if pos_i >= s_start: pos_i += len_s
            else: break
        
        # print(f'ID: {i}, Init.: {i}, Position: {pos_i}')

        if pos_i + len_scale < len(data):
            t_data = d_short[pos_i:pos_i+len_scale]
            m_data = np.mean(t_data)
            t_data = (t_data-m_data)*r_scale_t
            t_data += np.mean(data[pos_i:pos_i+len_scale])
            data_t[pos_i:pos_i+len_scale] = t_data
            label_t[pos_i:pos_i+len_scale] = 1
        else:
            count_t +=1
            print(f'Invalid position: {count_t} {i}=> {pos_i} / {len(data)}')
    
    # plot_anomaly(data_t, label_t)
    return data_t, label_t, d_inds

def add_local_anomaly_with_guard(data1, data2, d1_len, d2_len, win1, win2, p_scale, r_scale, guard_dur, scale_dist='uniform', dist_anomaly='uniform', keep_number=False):

    data1, data2 = data1.reshape(-1), data2.reshape(-1)
    if d1_len > d2_len:
        ## put anomalies (similar to d2) into d1
        d_long, d_short = data1, data2
        len_l, len_s = d1_len, d2_len
        len_scale = win2
    else:
        d_long, d_short = data2, data1
        len_l, len_s = d2_len, d1_len
        len_scale = win1
    data = np.array([])
    d_inds = np.array([])
    i =0
    while i < len(data1) and i< len(data2):
        data = np.append(data, d_short[i:i+len_s])
        d_inds = np.append(d_inds, np.ones(len(d_short[i:i+len_s])))
        i = i+len_s
        data = np.append(data, d_long[i:i+len_l])
        d_inds = np.append(d_inds, np.ones(len(d_long[i:i+len_l]))*2)
        i = i+len_l

    ## Num. of data2 
    cnt_n2 = len(d_inds[d_inds==2]) / d2_len

    ## computing noise range without guard duration
    add_noise_range = len(d_inds[d_inds==2]) - 2*math.floor(cnt_n2)*win2*guard_dur
    if len(d_inds[d_inds==2]) - d2_len*(math.floor(cnt_n2)) >= win2*guard_dur:
        add_noise_range -= win2*guard_dur

    data_t, label_t = copy.deepcopy(data), np.zeros(len(data))
    if p_scale <= 0: return data_t, label_t, d_inds
    
    num_anomalies = int(p_scale*len(data)/len_scale) if keep_number else int(p_scale*add_noise_range/len_scale)
    # print(f'Num. Anomalies: {num_anomalies}/{add_noise_range/len_scale} -- {num_anomalies/(add_noise_range/len_scale):.2f}')
    ind_scale = get_random_index(num_anomalies, add_noise_range, len_scale, dist_anomaly=dist_anomaly)
    small_intervals = find_anomaly_intervals(abs(d_inds-2))
    count_t = 0
    
    for i in ind_scale:
        if scale_dist == 'uniform':
            r_scale_t = random.uniform(r_scale/2, r_scale)
        elif scale_dist == 'Gaussian':
            r_scale_t = random.gauss(r_scale, 0.1)
        ## Find position to inject
        pos_i = i
        for (s_start, s_end) in small_intervals:
            ## add guard_dur at the start of n2
            if pos_i >= s_start: 
                if s_start == 0:
                    pos_i += len_s +win2*guard_dur
                else:
                    pos_i += len_s +2*win2*guard_dur
            elif pos_i > s_start - win2*guard_dur: 
                pos_i += len_s + 2*win2*guard_dur

        # print(f'ID: {i}, Init.: {i}, Position: {pos_i}')
        if pos_i + len_scale < len(data):
            t_data = d_short[pos_i:pos_i+len_scale]
            m_data = np.mean(t_data)
            t_data = (t_data-m_data)*r_scale_t
            t_data += np.mean(data[pos_i:pos_i+len_scale])
            data_t[pos_i:pos_i+len_scale] = t_data
            label_t[pos_i:pos_i+len_scale] = 1
        else:
            count_t +=1
            print(f'Invalid position: {count_t} {i}=> {pos_i} / {len(data)}')

    # plot_anomaly(data_t, label_t)
    return data_t, label_t, d_inds

def combine_ecg(data1, data2, label1, label2, d1_len, d2_len, win1, win2):

    data1, data2 = data1.reshape(-1), data2.reshape(-1)
    if d1_len > d2_len:
        ## put anomalies (similar to d2) into d1
        d_long, d_short = data1, data2
        l_long, l_short = label1, label2
        len_l, len_s = d1_len, d2_len
        # len_scale = win2
    else:
        d_long, d_short = data2, data1
        l_long, l_short = label2, label1
        len_l, len_s = d2_len, d1_len
        # len_scale = win1

    data, label = np.array([]), np.array([])
    d_inds = np.array([])
    i =0
    while i < len(data1) and i< len(data2):
        data = np.append(data, d_short[i:i+len_s])
        label = np.append(label, l_short[i:i+len_s])
        d_inds = np.append(d_inds, np.ones(len(d_short[i:i+len_s])))
        i = i+len_s
        data = np.append(data, d_long[i:i+len_l])
        label = np.append(label, l_long[i:i+len_l])
        d_inds = np.append(d_inds, np.ones(len(d_long[i:i+len_l]))*2)
        i = i+len_l

        
        # print(f'Elapsed: {i}/{max(len(data1), len(data2))}')

    return data, label, d_inds

## Inject anomalies using data_a. Put data_a (length of len_scale) into data
def add_shift_anomaly(data, label, data_a, p_scale, r_scale, len_scale, scale_dist='uniform', dist_anomaly='uniform'):
    data2, label2= copy.deepcopy(data), copy.deepcopy(label)
    len_data = len(data)
    ind_scale = get_random_index(int(p_scale*len_data/len_scale), len_data, len_scale, dist_anomaly=dist_anomaly)
    for i in ind_scale:
        if scale_dist == 'uniform':
            r_scale_t = random.uniform(r_scale/2, r_scale)
        elif scale_dist == 'Gaussian':
            r_scale_t = random.gauss(r_scale, 0.1)
            
        if i + len_scale < len(data):
            t_data = data_a[i:i+len_scale]
            m_data = np.mean(t_data)
            t_data = (t_data - m_data)*r_scale_t
            t_data += m_data
            data2[i:i+len_scale] = t_data
            label2[i:i+len_scale] = 1
        else:
            t_data = data[i:]
            m_data = np.mean(t_data)
            t_data = (t_data - m_data)*r_scale_t
            t_data += m_data
            data2[i:] = t_data
            label2[i:] = 1
            
    # plot_anomaly(data2, label2)
    return data2, label2

### Add selected ecg anomalies (ecg_anomaly) into data
def add_ecg_anomaly(data, label, idx, slidingWindow, ecg_anomaly):
    # idx_center_a = np.argmax(ecg_anomaly)
    idx_center_t = np.argmax(data[idx:idx+slidingWindow])
    t_seq = signal.resample(ecg_anomaly, slidingWindow)
    print('RESAMPLE:', len(t_seq))
    data[idx+idx_center_t-int(slidingWindow/2):idx+idx_center_t+int(slidingWindow/2)] = t_seq + data[int(idx+idx_center_t-int(slidingWindow/2))] - np.mean(t_seq)
    label[idx+idx_center_t-int(slidingWindow/2):idx+idx_center_t+int(slidingWindow/2)] = 1

    return data, label

## get clean ECG with alignment 
def split_normal_anomaly_ecg(data, label):
    data, label = data.reshape(-1), label.reshape(-1)
    l = find_length(data)
    print('Wind:', l)
    # data_n1 = np.array([])
    curr_start = 0
    a_inds = find_anomaly_intervals(label)

    to_merge = []
    for (anom_start, anom_end) in a_inds:
        curr_end = anom_start -l
        if curr_end - curr_start >= 2*l:
            to_merge.append(data[curr_start:curr_end])
        curr_start = anom_end + l

    d1 = to_merge[0]
    for i in range(len(to_merge)-1):
        t1 = d1[-2*l:]
        t2 = to_merge[i+1][:2*l]
        shift = np.argmax(np.correlate(t1, t2))
        if shift <=0:
            d1 = np.concatenate([d1, to_merge[i+1][abs(shift):]])
        else:
            d1 = np.concatenate([d1[:-shift], to_merge[i+1]])

    plt.figure(figsize=(12,2))
    plt.plot(d1)
    label_n1 = np.zeros(len(d1))

    data_a1, label_a1 = data[label==1], label[label==1].reshape(-1)
    return d1, data_a1

## get anomalies from ECG
def get_anomalies(data, label):
    anomalies, a_inds = [], []
    for (anom_start, anom_end) in find_anomaly_intervals(label.reshape(-1)):
        anomalies.append(data[anom_start:anom_end])
        # print(anom_start, anom_end)
        # print(np.arange(anom_start, anom_end))
        a_inds.append([anom_start,anom_end])
    return anomalies, a_inds

###################################################################################################################
## TSAD comparison

def compare_methods(data, label, slidingWindow, train_len, data_name, 
                    nm_len, overlap, kadj, normalize, selected_methods, stepwise=True, align=True, dist_org = True, cut=None, max_W = 20, REVISE_SCORE=True, delta =0, min_size=0.025, start_chunk=10000, chunk_size=5000, sp_index=1, x_lag=None, device_id=0):

    # MIN_SIZE_ON = 0.025
    if data_name == 'elec':
        slidingWindow = 48
    elif data_name ==  'weather':
        slidingWindow = 24
        
    print('SlidingWindow:', slidingWindow)

    ## for DAMP
    if sp_index <= slidingWindow:
        sp_index = slidingWindow +1
    if x_lag is not None:
        x_lag = 2**int(np.ceil(np.log2( x_lag*slidingWindow )))

    ## for SAND
    # start_chunk = 10000
    # chunk_size = 5000

    x_test = data
    scores = []
    slabels = []
    
    
    process_time = []

    ## for TranAD
    if 'TranAD' in selected_methods:
        args.dataset = 'Data'
        args.model = 'TranAD'
        args.retrain = True
        loader = []

        print('Test: TranAD')
        loader.append(x_test[:int(train_len)].reshape(int(train_len),1))
        loader.append(x_test.reshape(len(x_test),1))
        loader.append(label.reshape(len(label),1))
        train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
        test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
        label_tranad = loader[2]

        start_t = time()
        model, testD, testO, optimizer, scheduler = training_model(train_loader, test_loader, args.model, label_tranad, args.dataset)

        ### Testing phase
        torch.zero_grad = True
        model.eval()
        print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
        end_t = time()
        process_time.append(end_t-start_t)
        print('TranAD-Done (takes)', end_t - start_t)

        data_tranad = loader[1].reshape(len(loader[1]),)
        label_tranad = label_tranad.reshape(len(label_tranad),)
        score = loss
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        scores.append(score[:len(x_test)])
        slabels.append('TranAD (online)\nAnomaly Score')

        print(len(scores))

    if 'NormA' in selected_methods:
        print('Test: NORMA-OFF')
        # if normalize == 'z-norm': norma_norm = True
        # else: norma_norm= False
        # norma_norm = True
        normalize_comp = 'z-norm' if dist_org else normalize

        start_t = time()
        clf_off = NORMA(pattern_length = slidingWindow, nm_size=3*slidingWindow, percentage_sel=1, normalize=normalize_comp)
        clf_off.fit(x_test)
        end_t = time()
        process_time.append(end_t -start_t)
        print('NormA-Done (takes)', end_t - start_t)
        score = clf_off.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        scores.append(score[:len(x_test)])
        slabels.append('NormA (off)\nAnomaly Score')
    
        print(len(scores))
    if 'SAND' in selected_methods:
        modelName='SAND (online)'
        start_t = time()
        clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
        x = data
        clf.fit(x,online=True,alpha=0.5,init_length=start_chunk,batch_size=chunk_size,verbose=True,overlaping_rate=int(4*slidingWindow))
        end_t = time()
        process_time.append(end_t -start_t)
        print('SAND-Done (takes)', end_t - start_t)
        score = clf.decision_scores_
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        scores.append(score[:len(x_test)])
        slabels.append('SAND (online)\nAnomaly Score')

        print(len(scores))
    if 'DAMP' in selected_methods:
        modelName='DAMP'
        start_t = time()
        normalize_comp = 'z-norm' if dist_org else normalize
        clf = DAMP(m = slidingWindow,sp_index=sp_index, x_lag =x_lag, normalize=normalize_comp)
        x = data
        clf.fit(x)
        end_t = time()
        process_time.append(end_t -start_t)
        print('DAMP-Done (takes)', end_t - start_t)
        score = clf.decision_scores_
        score = running_mean(score, slidingWindow)
        score = np.array([score[0]]*math.ceil((slidingWindow-1)//2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        scores.append(score[:len(x_test)])
        slabels.append('DAMP (online)\nAnomaly Score')

        print(len(scores))

    if 'AnDri' in selected_methods:
        ## Offline
        modelName='AnDri (off)'
        print(f'R_min (off): {min_size}')
        start_t = time()
        clf = A2D2(pattern_length=slidingWindow, normalize=normalize, linkage_method='ward', th_reverse=5, kadj=kadj, nm_len=nm_len, overlap=overlap, max_W=max_W, eta=1, device_id=device_id)
        x = data
        clf.fit(x, y=label, online=False, training=True, training_len=int(train_len), stump=False, stepwise=stepwise, align=align, cut=cut, min_size=min_size)
        end_t = time()
        if len(clf.scores) == 0:
            num_min_cl_off = 0
            print(f'R_size (off): {num_min_cl_off}')
            scores.append(np.zeros(len(x_test)))
            
        else:    
            num_min_cl_off = len(clf.listcluster[clf.listcluster == -1])
            print(f'R_size (off): {num_min_cl_off}')
            score = clf.scores
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) < len(x):
                score = np.append(score, np.ones(len(x)-len(score))*np.mean(score))
            scores.append(score[:len(x_test)])
        slabels.append('AnDri (off)')
        process_time.append(end_t -start_t)
        print('AnDri-Done (takes)', end_t - start_t)
        
        ## Online
        modelName='AnDri (on)'
        print(f'R_min (on): {min_size}')
        start_t = time()
        clf = A2D2(pattern_length=slidingWindow, normalize=normalize, linkage_method='ward', th_reverse=5, kadj=kadj, nm_len=nm_len, overlap=overlap, max_W=max_W, eta=1, device_id=device_id, REVISE_SCORE=REVISE_SCORE)
        x = data
        # min_on = MIN_SIZE_ON if len(data)*min_size > train_len*MIN_SIZE_ON else min_size
        # min_on = len(data)/train_len*min_size
        # print(f'R_MIN (online): {min_on} vs. {min_size}')
        min_on = min_size
        clf.fit(x, y=label, online=True, training=True, training_len=int(train_len),  delta=delta, stump=False, stepwise=stepwise, align=align, cut=cut, min_size=min_on)
        end_t = time()
        if len(clf.scores) ==0 :
            num_min_cl_on = 0
            print(f'R_size (on): {num_min_cl_on}')
            scores.append(np.zeros(len(x_test)))
        else:
            num_min_cl_on = len(clf.listcluster[clf.listcluster == -1])
            print(f'R_size (on): {num_min_cl_on}')
            score = clf.scores
            score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
            if len(score) < len(x):
                score = np.append(score, np.ones(len(x)-len(score))*np.mean(score))
            scores.append(score[:len(x_test)])
        slabels.append('AnDri (on)')
        process_time.append(end_t -start_t)
        print('AnDri-Done (takes)', end_t - start_t)
        
        
        
        # slabels.append('A2D2 (on) Flatten')
        print(len(scores))

    time_df = pd.DataFrame(process_time, columns=['time'])
    # display(time_df)
    return scores, slabels, time_df, num_min_cl_off, num_min_cl_on

def result_acc(methods, scores, label, slidingWindow):
    result_org = pd.DataFrame(columns=['method'] + rst_columns)
    j = 0
    for i, method in enumerate(methods):
        r_tmp = get_acc(label.reshape(-1)[:len(scores[i])], np.array(scores[i]), slidingWindow)
        result_org.loc[j] = [method] + list(r_tmp.loc[0])
        j+=1

    display(result_org)
    return result_org

def find_best_f1(score, label, chk=False):
    p, r, f, t = [], [], [], []
    # th_max = np.round(0.95, 1)
    # th_min = np.round(0.05, 1)
    
    th_min, th_max = 0.05, 0.95
    # print('START: ', th_min, 'to', th_max)
    for ths in np.arange(th_min, th_max, (th_max-th_min)/10):
        preds = score > ths
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        p.append(Precision[1])
        r.append(Recall[1])
        f.append(F[1])
        t.append(ths)
    if chk:
        return p, r, f, t

    id_f = np.argmax(f)
    if id_f == 0:
        # print('Something Wrong', id_f, f[id_f], t[id_f])
        ths = 0.99
        preds = score> ths
        _, _, F, _ = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        th_r = [ths, t[id_f+1]]
        f_r = [F[1], f[id_f+1]]
        # return p[id_f], r[id_f], f[id_f], t[id_f]
    if id_f == len(f)-1:
        # print('Something Wrong', id_f, f[id_f], t[id_f])
        ths = 0.0001
        preds = score> ths
        _, _, F, _ = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        th_r = [ths, t[id_f-1]]
        f_r = [F[1], f[id_f-1]]
        # return p[id_f], r[id_f], f[id_f], t[id_f]
    else:
        # print('chk:', len(f), id_f)
        if f[id_f-1] > f[id_f+1]:
            th_r = [t[id_f-1], t[id_f]]
            f_r = [f[id_f-1], f[id_f]]
        else:
            th_r = [t[id_f], t[id_f+1]]
            f_r = [f[id_f], f[id_f+1]]

    # print(f_r, th_r)
    while abs(f_r[0] - f_r[1]) > 0.001:
        ths = np.mean(th_r)
        preds = score > ths
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        if F[1] < np.min(f_r):
            ths = th_r[0] if f_r[0] > f_r[1] else th_r[1]
            preds = score > ths
            # print('[1] ths', ths, th_r, f_r)
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            break
        
        th_r = [ths, th_r[np.argmax(f_r)]]
        f_r = [F[1], np.max(f_r)]
        
        # print(f_r, th_r)
        if abs(f_r[0]-f_r[1]) < 0.001:
            ths = th_r[0] if f_r[0] > f_r[1] else th_r[1]
            preds = score > ths
            Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
            break

    return Precision[1], Recall[1], F[1], ths
