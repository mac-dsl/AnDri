import numpy as np
import stumpy
import copy
import pickle
# import torch

from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean, cdist
from tslearn.metrics import dtw
from numpy.fft import fft, ifft
from numpy.linalg import norm, eigh

from scipy.signal import argrelextrema, correlate

from statsmodels.tsa.stattools import acf
from tslearn.utils import to_time_series_dataset,to_time_series
from tslearn.metrics.cycc import cdist_normalized_cc, y_shifted_sbd_vec
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
########################################################################################

############################################################################
## Code from TSB-UAD
## https://github.com/TheDatumOrg/TSB-UAD
## sand.py
def ncc_c(x, y):
	den = np.array(norm(x) * norm(y))
	den[den == 0] = np.inf #np.Inf

	x_len = len(x)
	fft_size = 1 << (2*x_len-1).bit_length()
	cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
	cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
	return np.real(cc) / den

def sbd(x, y):
    ncc = ncc_c(x, y)
    idx = ncc.argmax()
    # print('CHK', len(x), len(y), len(ncc), idx)
    dist = 1 - ncc[idx]
    return dist

# Computation of the updated centroid
def extract_shape_stream(X,cluster_centers):
	X = to_time_series_dataset(X)
	cluster_centers = to_time_series(cluster_centers)
	sz = X.shape[1]
	Xp = y_shifted_sbd_vec(cluster_centers, X,
					norm_ref=-1,
					norms_dataset=np.linalg.norm(X, axis=(1, 2)))
	S = np.dot(Xp[:, :, 0].T, Xp[:, :, 0])
	# if not initial:    
		# S = S + self.S[idx]
	# self.S[idx] = S
	Q = np.eye(sz) - np.ones((sz, sz)) / sz
	M = np.dot(Q.T, np.dot(S, Q))
	_, vec = np.linalg.eigh(M)
	mu_k = vec[:, -1].reshape((sz, 1))
	dist_plus_mu = np.sum(np.linalg.norm(Xp - mu_k, axis=(1, 2)))
	dist_minus_mu = np.sum(np.linalg.norm(Xp + mu_k, axis=(1, 2)))
	if dist_minus_mu < dist_plus_mu:
		mu_k *= -1
    # return mu_k
	return mu_k # _zscore(mu_k, ddof=1),S
    

########################################################################################
def divide_subseq(seq, slidingWindow, r, overlap=0.5, label=None):
    nm_size = round(slidingWindow*r)
    step = round(nm_size*(1-overlap))
    if len(seq) < nm_size: return None
    subseq = []
    for i in range(0, len(seq)-nm_size, step): subseq.append([i, i+nm_size])

    aligned_idx = _unshift_series(seq, subseq, nm_size)
    rev_idx= []
    for value in aligned_idx:
         if value not in rev_idx: rev_idx.append(value)

    result, result_label = [], []
    for s_e in rev_idx:
        result.append(seq[s_e[0]:s_e[1]])
        if label is not None: result_label.append(label[s_e[0]:s_e[1]])

    if label is None: return result
    else: return result, result_label

## To use 'SBD', you have to divide subsequences with the length of self.pattern_length
def norm_seq(seq, sel='zero-mean'):
    if sel == 'z-norm':
        if np.std(seq) ==0: t_std = 0.000001
        else: t_std = np.std(seq)
        seq_n = (seq - np.mean(seq))/t_std
    elif sel == 'zero-mean':
        seq_n = seq - np.mean(seq)
    elif sel == 'z-norm_rev':
        seq_n = (seq - np.mean(seq)/(1+np.std(seq)))
    elif sel == 'euclidean' or 'sbd' or 'dtw':
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
    return d_v

def compute_seq_dist(seq_l, seq_s, metric='euclidean'):
    if len(seq_l) < len(seq_s):
        seq_n  = copy.deepcopy(seq_s)
        seq_s = seq_l
        seq_l = seq_n

    win_len = len(seq_s)
    dist = []
    if metric in ['euclidean', 'zero-mean', 'z-norm', 'z-norm_rev']:
        if len(seq_l) == len(seq_s):
            d_v = np.array(seq_l) - np.array(seq_s)
            return np.linalg.norm(d_v)
        else:
            for i in range(len(seq_l)-len(seq_s)):
                t_d = np.array(seq_l[i:i+win_len]) - np.array(seq_s)
                dist.append(np.linalg.norm(t_d))
            idx = np.argmin(dist)
            d_v = np.array(seq_l[idx:idx+win_len]) - np.array(seq_s)
            return np.linalg.norm(d_v)
    elif metric == 'sbd':
        return sbd(seq_l, seq_s)
    elif metric == 'dtw':
        return dtw(seq_l, seq_s)
    elif metric == 'fastDTW':
        d_v, _ = fastdtw(seq_l, seq_s, dist=lambda a, b: abs(a - b))
        return d_v
    
## d_subseq should be normalized distance if normalize=True
def intra_cluster_dist_stat(d_subseq):
    num_cl = len(d_subseq)
    d_ci, d_c_std = [],[]
    for i in range(num_cl):
        # if len(d_subseq[i]) < 2: continue

        # d_t = [np.linalg.norm(seq)/2 for seq in d_subseq[i]]
        d_t = d_subseq[i]
        ## take only top 95%
        d_t.sort()
        d_ci.append(np.mean(d_t))
        d_c_std.append(np.std(d_t))

    return np.array(d_ci), np.array(d_c_std) #, th_ci

def intra_cluster_dist(d_subseq):
    num_cl = len(d_subseq)
    d_ci, d_c_std = [],[]
    for i in range(num_cl):
        if len(d_subseq[i]) < 2: continue

        d_t = [np.linalg.norm(seq)/2 for seq in d_subseq[i]]
        # d_t = d_subseq
        ## take only top 95%
        d_t.sort()
        d_ci.append(np.mean(d_t))
        d_c_std.append(np.std(d_t))

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

def offline_score(ts, pattern_length, nms, cl_s, normalize, use_GPU=False):
    # Compute score
    all_join = []
    for index_name in range(len(nms)):   
        # if torch.cuda.is_available() and use_GPU:
            # join = stumpy.gpu_stump(ts,pattern_length,nms[index_name].subseq,ignore_trivial = False, normalize=normalize)[:,0]
        # else:
        join = stumpy.stump(ts,pattern_length,nms[index_name].subseq,ignore_trivial = False, normalize=normalize)[:,0]
        join = np.array(join)
        all_join.append(join)

    for i in range(len(all_join[0])):
        if cl_s[i] >=0: 
            join[i] = all_join[int(cl_s[i])][i]/nms[int(cl_s[i])].tau
        else:
            tmps = []
            for a_j in all_join: tmps.append(a_j[i])
            join[i] = np.min(tmps)/nms[np.argmin(tmps)].tau

    join = np.array([join[0]]*(pattern_length//2) + list(join) + [join[-1]]*(pattern_length//2))        
    join_n = running_mean(join,pattern_length)
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

def backward_anomaly_step(ts, pattern_length, nm_subseq, normalize, device_id=0, use_GPU=False):
    score = []
    ts_n = norm_seq(ts, normalize)
    if normalize == 'euclidean' or normalize == 'z-norm' or normalize == 'zero-mean' or normalize == 'z-norm_rev':
        # if torch.cuda.is_available() and use_GPU:
            # score = stumpy.gpu_stump(ts_n, pattern_length, nm_subseq, device_id = device_id, ignore_trivial=False, normalize=False)[:,0]
        # else:
        score = stumpy.stump(ts_n, pattern_length, nm_subseq, ignore_trivial=False, normalize=False)[:,0]
    else:
        for i in range(len(ts)-pattern_length):
            tmp_x = ts_n[i:i+pattern_length]
            if normalize == 'sbd':
                score.append(sbd(nm_subseq, tmp_x))
            elif normalize == 'dtw':
                score.append(dtw(nm_subseq, tmp_x)) 
            elif normalize == 'fastDTW':
                d_v, _ = fastdtw(nm_subseq, tmp_x, dist=lambda a, b: abs(a - b))
                score.append(d_v)
            else:
                print("Not supported normalization method")
                break
    score = np.array(score)
    return score

def backward_anomaly_changing_point(ts, pattern_length, nms, normalize, device_id=0, use_GPU=False):
    scores = []
    ts_n = norm_seq(ts, normalize)
    
    for nm in nms:
        # if torch.cuda.is_available() and use_GPU:
            # scores.append(stumpy.gpu_stump(ts_n, pattern_length, nm.subseq, device_id=device_id,ignore_trivial=False, normalize=False)[:,0])
        # else:
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
    # rev_scores = running_mean(rev_scores, slidingWindow)
    # rev_scores = np.array([rev_scores[0]]*((slidingWindow-1)//2) + list(rev_scores) + [rev_scores[-1]]*((slidingWindow-1)//2))

    return rev_scores

def read_pickle(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def write_pickle(f_name, data):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return True
    
