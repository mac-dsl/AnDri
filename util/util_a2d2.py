import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stumpy
import sys
import copy
import pickle
import torch


from numpy.fft import fft, ifft
# from util.TSB_AD.metrics import metricor
# import matplotlib.patches as mpatches 

from scipy.signal import argrelextrema, correlate
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, distance

from statsmodels.tsa.stattools import acf

############################################################################
## Code from TSB-UAD
## https://github.com/TheDatumOrg/TSB-UAD
## slidingWindows.py
## "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection"
## John Paparrizos, Yuhao Kang, Paul Boniol, Ruey Tsay, Themis Palpanas, and Michael Franklin.
## Proceedings of the VLDB Endowment (PVLDB 2022) Journal, Volume 15, pages 1697–1711

# Function to find the length of period in time-series data
def find_length(data):
    """
    Finds the length of the period in time-series data.

    Args:
    - data: Time-series data, ndarray

    Returns:
    - Length of period: Integer
    """
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3: #or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        # return 125
        return 300
    
###########################################################################
####### Align the recurrent sequences #######
def _unshift_series(ts, sequence_rec,normalmodel_size):
	result = []
	ref = ts[sequence_rec[0][0]:sequence_rec[0][1]]
	for seq in sequence_rec:
		shift = (np.argmax(correlate(ref, ts[seq[0]:seq[1]])) - len(ts[seq[0]:seq[1]]))
		if (len(ts[seq[0]-int(shift):seq[1]-int(shift)]) == normalmodel_size):
			result.append([seq[0]-int(shift),seq[1]-int(shift)])
	return result


# SBD distance
def __sbd(x, y):
	ncc = __ncc_c(x, y)
	idx = ncc.argmax()
	dist = 1 - ncc[idx]
	return dist, None

def __ncc_c(x, y):
	den = np.array(norm(x) * norm(y))
	den[den == 0] = np.Inf
	x_len = len(x)
	fft_size = 1 << (2*x_len-1).bit_length()
	cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
	cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
	return np.real(cc) / den
########################################################################################

########################################################################################
def divide_subseq(seq, slidingWindow, r, overlap=0.5, label=None):
    nm_size = round(slidingWindow*r)
    step = round(nm_size*(1-overlap))
    if len(seq) < nm_size:
        return None
    # print('Stepsize:', step)
    subseq = []
    for i in range(0, len(seq)-nm_size, step):
        # subseq.append(seq[i:i+slidingWindow])
        subseq.append([i, i+nm_size])
    # print('Num of subseq:', len(subseq), i)

    aligned_idx = _unshift_series(seq, subseq, nm_size)
    rev_idx= []
    for value in aligned_idx:
         if value not in rev_idx:
              rev_idx.append(value)


    result, result_label = [], []
    for s_e in rev_idx:
        result.append(seq[s_e[0]:s_e[1]])
        if label is not None:
            result_label.append(label[s_e[0]:s_e[1]])

    if label is None:
        return result
    else:
        return result, result_label

## To use 'SBD', you have to divide subsequences with the length of self.pattern_length
def norm_seq(seq, sel='zero-mean'):
    if sel == 'z-norm':
        if np.std(seq) ==0: t_std = 0.000001
        else: t_std = np.std(seq)
        seq_n = (seq - np.mean(seq))/t_std
    elif sel == 'zero-mean':
        seq_n = seq - np.mean(seq)
    elif sel == 'euclidean' or 'sbd':
        seq_n = seq
    return seq_n

def compute_diff_dist(seq_l, seq_s):
    if len(seq_l) < len(seq_s):
        seq_n = copy.deepcopy(seq_s)
        seq_s = seq_l
        seq_l = seq_n

    win_len = len(seq_s)
    dist = []
    if len(seq_l) == len(seq_s):
        d_v = np.array(seq_l)-np.array(seq_s)
        return d_v
    for i in range(len(seq_l)-len(seq_s)+1):
        t_d = np.array(seq_l[i:i+win_len]) - np.array(seq_s)
        dist.append(np.linalg.norm(t_d))

    idx = np.argmin(dist)
    d_v = np.array(seq_l[idx:idx+win_len]) - np.array(seq_s)
    # return np.min(dist), np.argmin(dist), dist
    return d_v

## d_subseq should be normalized distance if normalize=True
def intra_cluster_dist(d_subseq):
    num_cl = len(d_subseq)
    d_ci, d_c_std = [],[]
    # print('# of CL: ', num_cl)
    for i in range(num_cl):
        if len(d_subseq[i]) < 2:
            continue
        # d_t = [abs(seq) for seq in d_subseq[i]]
        # # print('CL', i, 'has', len(d_t), 'members')
        # d_ci.append(np.sum(d_t, axis=1))
        # d_c_std.append(np.std(d_t, axis=1))
        d_t = [np.linalg.norm(seq)/2 for seq in d_subseq[i]]
        ## take only top 95%
        d_t.sort()
        # d_t = d_t[:int(len(d_t)*0.95)]
        d_ci.append(np.mean(d_t))
        d_c_std.append(np.std(d_t))

    # th_ci = [(np.mean(d)+3*np.std(d))/2 for d in d_ci]

    return np.array(d_ci), np.array(d_c_std) #, th_ci

def ewma(m, std, val, span=10):
    alpha = 2/(1+span)
    diff = val - m
    incr = alpha*diff
    new_m = m + incr
    new_var = (1-alpha) * (std + diff*incr)
    return new_m, np.sqrt(new_var)

def longest_consecutive_sequence(lst):
    if not lst:
        return 0

    max_length = 1
    current_length = 1
    
    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1
    
    return max_length

def running_mean(x,N):
	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N

def offline_score(ts, pattern_length, nms, cl_s, normalize):
    # Compute score
    # ts_n = norm_seq(ts, normalize)
    all_join = []
    for index_name in range(len(nms)):   
        if torch.cuda.is_available():
            join = stumpy.gpu_stump(ts,pattern_length,nms[index_name].subseq,ignore_trivial = False, normalize=normalize)[:,0]
        else:
            join = stumpy.stump(ts,pattern_length,nms[index_name].subseq,ignore_trivial = False, normalize=normalize)[:,0]
        join = np.array(join)
	    #join = (join - min(join))/(max(join) - min(join))
        all_join.append(join)

    # join = np.min(all_join, axis=0)
    for i in range(len(all_join[0])):
        if cl_s[i] >=0: 
            join[i] = all_join[int(cl_s[i])][i]/nms[int(cl_s[i])].tau
        else:
            tmps = []
            for a_j in all_join: tmps.append(a_j[i])
            join[i] = np.min(tmps)/nms[np.argmin(tmps)].tau

    join = np.array([join[0]]*(pattern_length//2) + list(join) + [join[-1]]*(pattern_length//2))        
    join_n = running_mean(join,pattern_length)
    # join_n = join
    #reshifting the score time series
    join_n = np.array([join_n[0]]*(pattern_length//2) + list(join_n) + [join_n[-1]]*(pattern_length//2))
    return join_n, all_join

def backward_anomaly(ts, pattern_length, nm_subseq, normalize):
    score = []
    for i in range(len(ts)-pattern_length):
        tmp_x = ts[i:i+pattern_length]
        tmp_x_n = norm_seq(tmp_x, normalize)
        score.append(np.linalg.norm(compute_diff_dist(nm_subseq, tmp_x_n)))
    return np.array(score)

def backward_anomaly2(ts, pattern_length, nm_subseq, normalize, device_id=0):
    score = []
    ts_n = norm_seq(ts, normalize)
    if torch.cuda.is_available():
        score = stumpy.gpu_stump(ts_n, pattern_length, nm_subseq, device_id = device_id, ignore_trivial=False, normalize=False)[:,0]
    else:
        score = stumpy.stump(ts_n, pattern_length, nm_subseq, ignore_trivial=False, normalize=False)[:,0]
    score = np.array(score)
    return score

def backward_anomaly_changing_point(ts, pattern_length, nms, normalize, device_id=0):
    scores = []
    ts_n = norm_seq(ts, normalize)
    
    for nm in nms:
        if torch.cuda.is_available():
            scores.append(stumpy.gpu_stump(ts_n, pattern_length, nm.subseq, device_id=device_id,ignore_trivial=False, normalize=False)[:,0])
        else:
            scores.append(stumpy.stump(ts_n, pattern_length, nm.subseq, ignore_trivial=False, normalize=False)[:,0])
    score = np.min(scores, axis=0)

    score = np.array(score)
    return score

def align_score(NMs, scores, cl_s, slidingWindow):
    num_nm = len(NMs)
    rev_scores = copy.deepcopy(scores[:len(cl_s)])
    ## for each NM,
    for i in range(num_nm):
        tmp_sc = rev_scores[cl_s ==i]
        if len(tmp_sc) >0:
            tmp_sc = tmp_sc - np.mean(tmp_sc) + 0.1
            rev_scores[cl_s ==i] = tmp_sc
    rev_scores[rev_scores <0] = 0
    rev_scores = running_mean(rev_scores, slidingWindow)
    rev_scores = np.array([rev_scores[0]]*((slidingWindow-1)//2) + list(rev_scores) + [rev_scores[-1]]*((slidingWindow-1)//2))

    return rev_scores

def read_pickle(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def write_pickle(f_name, data):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return True
    
