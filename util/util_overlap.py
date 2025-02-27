import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import stumpy
import sys

from util.TSB_AD.metrics import metricor
import matplotlib.patches as mpatches 

from scipy.signal import argrelextrema, correlate
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, distance

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
# Class representing a sliding window for anomaly detection
class window_L():
    def __init__(self, size, NMs=[], ths=[], th_a=[], Ws=[], Fs=[], th_drift=None, normalize='zero-mean'):
        self.size = size        
        # self.seq = []   ## sequences are retrieved from streaming X (for size of W_L)
        self.cl = []
        self.label = []
        self.score = []

        # self.active_NMs = NMs
        self.active_IDs = [i for i in range(len(NMs))]
        # self.inactive_NMs = []
        self.inactive_IDs = []
        # self.thresholds = ths
        # self.th_a = th_a
        self.Ws = Ws
        self.Fs = Fs
        self.curr = 0
        self.w_L = np.max(Ws)
        self.normalize = normalize
        if th_drift is None:
            self.th_drift = np.min(ths)
        else:
            self.th_drift = th_drift
        # self.counters = Fs 

    # Method to add a new normal model to the sliding window
    def add_NM(self, W, F):
        self.active_IDs.append(len(self.active_IDs))
        # self.thresholds.append(ths)
        # self.th_a.append(th_a)
        self.Ws = np.append(self.Ws,W)
        self.Fs = np.append(self.Fs, F)
        self.w_L = np.max(self.Ws)

    def enqueue(self, label, cl, score):
        self.label = np.append(self.label, label)
        self.cl = np.append(self.cl, cl)
        self.score = np.append(self.score, score)
        self.curr +=1

        if self.curr > self.size:
            self.label = np.delete(self.label, 0)
            self.cl = np.delete(self.cl, 0)
            self.score = np.delete(self.score, 0)
            self.curr = self.size

        if label ==0:  return 4 if (self._examine_inactive()) else 0
        else:
            ## TODO: Revise here!!! (current cl is only in active_IDs)
            ## Compare it from inactive_IDs first and update it on the queue
            if cl in self.inactive_IDs:
                # print('CHECK HIT')
                return 5 if (self._examine_active(cl)) else 1
            else:
                # print('Anomaly detected')
                return 2

    # Method to force active NMs to inactive (Normals)
    def _examine_inactive(self):  
        avg_w = round(np.mean(self.Ws))
        for id in self.active_IDs:
            ## at the start (only smnall set is in)
            if self.Ws[id] > len(self.cl): continue
            w_id = self.cl[int(-self.Ws[id]):]

            # test_Ws = self.Ws[id] if self.Ws[id] > avg_w else avg_w
            # if test_Ws > len(self.cl): continue            
            # w_id = self.cl[-int(test_Ws):]
            
            # if len([c for c in w_id if c==id]) < self.Fs[id]:
            if len([c for c in w_id if c==id]) ==0:
                ## Force id to inactive
                print('[INACTIVE] Cluster', id, 'is inactive', len([c for c in w_id if c==id]), 'chk:', w_id)                
                # print('chk', w_id)
                self.inactive_IDs.append(id)
                self.active_IDs.remove(id)
                print('Active:', self.active_IDs, 'Inactive:', self.inactive_IDs)
                return True
        return False
         
    ## To force inactive NMs to active (Re-occurring drift)
    ## TODO: Check it works well w/ re-occurring drift 
    def _examine_active(self, cl):
        w_id = self.cl[int(-self.Ws[cl]):]
        # print([c for c in w_id if c==cl], 'vs.', self.Fs[cl])
        if len([c for c in w_id if c==cl]) > self.Fs[cl]:
            print('[Re-occurring] Cluster', cl , 'is now active', len([c for c in w_id if c==cl]))
            self.active_IDs.append(cl)
            self.inactive_IDs.remove(cl)
            return True
        return False

    ## To check new NM as a drift         
    ## TODO: This condition does not seem to work (active / inactive / new normal)
    def examine_anomalies(self, seq, slidingWindow):
        ## No anomalies in w_L
        # if not (1 in self.label[:-1]): return None, 0, 0, []

        # idx = [i for i in range(len(self.label)-1) if self.label[i]==1]
        idx = [i for i in range(len(self.label)) if self.label[i]==1 and self.cl[i] in self.active_IDs]
        if len(idx) < np.mean(self.Fs):    return None, 0, 0, []
        print('COMPARE',[self.cl[i] for i in range(len(self.label)) if self.label[i] ==1], 'to', idx)
        
        curr_a = seq[-slidingWindow:]
        cnt_repeat = 0
        sim_anomaly = []

        ## TODO: Revise the mechanism to find new normal
        # samples = []
        ids_sample = []
        for id in idx: ## index of anomalies            
            # seq_t = seq[:(2)*slidingWindow] if id ==0 else seq[((id-1)*slidingWindow):(id+1)*slidingWindow]
            # seq_t_n = (seq_t - np.mean(seq_t))/np.std(seq_t) if self.normalize else (seq_t -np.mean(seq_t))
            # samples.append(seq_t_n)
            if id == 0:
                start_i = 0
                end_i = 2*slidingWindow
            else:
                start_i = (id-1)*slidingWindow
                end_i = (id+1)*slidingWindow
            ids_sample.append([start_i, end_i])
        
        # print('CHK TS:', len(seq), ids_sample)
        aligned_idx = _unshift_series(seq, ids_sample, slidingWindow*2)
        # print('ALIGNED:', aligned_idx)
        re_samples = []
        for s_e in aligned_idx: 
            seq_t = seq[s_e[0]:s_e[1]]
            # seq_t_n = (seq_t - np.mean(seq_t))/np.std(seq_t) if self.normalize else (seq_t -np.mean(seq_t))
            seq_t_n = norm_seq(seq_t, self.normalize)
            re_samples.append(seq_t_n)

        # d_samples = distance.pdist(samples)
        # d_samples = distance.squareform(d_samples)
        # for i in range(len(d_samples)): d_samples[i][i] = 10000
        # print('CHK_CONT_ANOMALIES', len(idx), len(d_samples))
        # print(d_samples)
        # for i, ds in enumerate(d_samples):
        #     print('CHK:', i, ds)
        #     sel_is = [sel_i for sel_i,d in enumerate(ds) if d<self.th_drift]
        #     if len([d for d in ds if d<self.th_drift]) > np.mean(self.Fs):
        #         ## Found!
        #         print('New Normal model is needed', len(sel_is), np.mean(self.Fs), self.th_drift)

        # print('??',len(re_samples))
        re_d_samples = distance.pdist(re_samples)
        re_d_samples = distance.squareform(re_d_samples)
        for i in range(len(re_d_samples)): re_d_samples[i][i] = 10000
        # print('[SHIFT] CHK_CONT_ANOMALIES', len(idx), len(re_d_samples))
        # print(re_d_samples)
        for i, ds in enumerate(re_d_samples):
            # print('CHK:', i, ds)
            sel_is = [sel_i for sel_i,d in enumerate(ds) if d<(self.th_drift)/2]
            if len(sel_is) > np.mean(self.Fs):
                ## Found!
                print('[NEW NORMAL] New Normal model is needed', [d for d in ds if d<self.th_drift], 'ID:', sel_is, '>', np.mean(self.Fs), f'TH: {self.th_drift}')
                t_subseq = [re_samples[k] for k in sel_is]
                # plt.figure(figsize=(10,3))
                # [plt.plot(tsub) for tsub in t_subseq]
                # plt.show()
                candidate_seq = np.mean(t_subseq, axis=0)
                c_intra_dist = [candidate_seq - np.array(k_subseq) for k_subseq in t_subseq]
                new_d_ci,_ ,_ = intra_cluster_dist([c_intra_dist])
                W_n = self.size
                F_n = round(len(t_subseq)/2)+1
                # W_n = (np.max(cont_idx)+1)
                # F_n = np.mean(cont_idx)
                print('New W and F:', W_n, F_n)
                # print(candidate_seq)
                return candidate_seq, W_n, F_n, new_d_ci
            
        return None, 0, 0, []


        # for id in idx:
        #     ## get anomaly seq. with 1.5x slindingWindow (from -1/2~1)
        #     seq_t = seq[:(id+1)*slidingWindow+int(slidingWindow)] if id ==0 else seq[(id*slidingWindow-int(slidingWindow)):(id+1)*slidingWindow]
        #     
        #     if self.normalize:
        #         seq_t_n = (seq_t - np.mean(seq_t))/np.std(seq_t)
        #         curr_a_n = (curr_a - np.mean(curr_a))/np.std(curr_a)
        #     else:
        #         seq_t_n = (seq_t - np.mean(seq_t))
        #         curr_a_n = (curr_a - np.mean(curr_a))
        #     d_v = compute_diff_dist(seq_t_n, curr_a_n)
        #     if np.sum(abs(d_v)) < self.th_drift: 
        #         cnt_repeat += 1
        #         sim_anomaly.append(id)
        #     print('chk2', id, 'size:', int(self.size), 'dist:', np.sum(abs(d_v)), 'TH:', self.th_drift, cnt_repeat, 'for', np.mean(self.Fs))

        #if cnt_repeat > np.mean(self.Fs):             
        #    print('New Normal model is needed', cnt_repeat, np.max(self.Fs), self.th_drift)
        #    ## Add a new normal model w/ 2x slidingWindow
        #    cont_idx = np.array(sim_anomaly[1:]) - np.array(sim_anomaly[:-1])
        #    chk_idx = [i for i, id in enumerate(cont_idx) if id ==1]    ## Try to find the continuous similar sequences
        #    print('IDs:', chk_idx)
        #    if len(chk_idx) >=1:
        #        candidate_idx = [sim_anomaly[id] for id in chk_idx]
        #        candidates = [seq[id*slidingWindow:id*slidingWindow+2*slidingWindow] for id in candidate_idx]            
        #    else:
        #        candidates = [seq[id*slidingWindow:id*slidingWindow+2*slidingWindow] for id in sim_anomaly]
#
        #    if self.normalize:
        #        t_subseq = [(c_seq - np.mean(c_seq))/np.std(c_seq) for c_seq in candidates]
        #    else:
        #        t_subseq = [(c_seq - np.mean(c_seq)) for c_seq in candidates]
        #    
        #    candidate_seq = np.mean(t_subseq,axis=0)
        #    c_intra_dist = [candidate_seq - np.array(i_subseq) for i_subseq in t_subseq]
        #    new_d_ci,_ ,_ = intra_cluster_dist([c_intra_dist])
#
        #    ## Set W and F TODO: verify it!!!
        #    W_n = self.size
        #    F_n = round(len(sim_anomaly)/2)
        #    # W_n = (np.max(cont_idx)+1)
        #    # F_n = np.mean(cont_idx)
        #    print('New:', W_n, F_n, 'sel_ID:', sim_anomaly)
        #    ## TODO: After adding a new normal model, clear window_L? (for new NMs)
        #    ## Need to keep updating the new normal model
#
        #    return candidate_seq, W_n, F_n, new_d_ci
        #
        #else:
        #    return None, 0, 0, []

    
###########################################################################
def get_anomaly_subseq(x_train, label, slidingWindow):
    # train_anomaly = x_train[label ==1]
    ind_anomaly = [x for x in range(len(label)) if label[x] == 1]
    # train_normal_p = x_train[label == 0]
    # print('MEAN:', np.mean(x_train), np.mean(train_anomaly), np.mean(train_normal_p))
    # print('STD:', np.std(x_train), np.std(train_anomaly), np.std(train_normal_p))

    if len(ind_anomaly) == 0: return [], [x_train], []
    
    anomaly_seq = []
    anomaly_label = []
    ind_s, ind_e = [], []
    s, e = 0, 0
    # print(ind_anomaly)
    for i in ind_anomaly:
        s = i-round(slidingWindow/2)
        if s > 0 and s > e-(slidingWindow //2):
            e = s+slidingWindow
            seq = x_train[s:e]
            label_a = label[s:e]
            anomaly_seq.append(seq)
            anomaly_label.append(label_a)
            ind_s.append(s)
            ind_e.append(e)
            
    normal_seq = []
    for i in range(len(ind_s)+1):
        if i ==0: seq_t = x_train[:ind_s[0]]
        elif i == len(ind_s): seq_t = x_train[ind_e[i-1]:]
        else: seq_t = x_train[ind_e[i-1]:ind_s[i]]

        if len(seq_t) >= slidingWindow:
            normal_seq.append(seq_t)

    return anomaly_seq, normal_seq, anomaly_label

####### Align the recurrent sequences #######
def _unshift_series(ts, sequence_rec,normalmodel_size):
	result = []
	ref = ts[sequence_rec[0][0]:sequence_rec[0][1]]
	for seq in sequence_rec:
		shift = (np.argmax(correlate(ref, ts[seq[0]:seq[1]])) - len(ts[seq[0]:seq[1]]))
		if (len(ts[seq[0]-int(shift):seq[1]-int(shift)]) == normalmodel_size):
			result.append([seq[0]-int(shift),seq[1]-int(shift)])
	return result

def divide_subseq(seq, slidingWindow, r, label=None):
    step = round(slidingWindow*r)
    if len(seq) < step:
        return []
    # print('Stepsize:', step)
    subseq = []
    for i in range(0, len(seq), step):
        # subseq.append(seq[i:i+slidingWindow])
        subseq.append([i, i+step])
    print('Num of subseq:', len(subseq), i)

    aligned_idx = _unshift_series(seq, subseq, step)
    result, result_label = [], []
    for s_e in aligned_idx:
        result.append(seq[s_e[0]:s_e[1]])
        if label is not None:
            result_label.append(label[s_e[0]:s_e[1]])

    if label is None:
        return result
    else:
        return result, result_label

def norm_seq(seq, sel='zero-mean'):
    if sel == 'z-norm':
        if np.std(seq) ==0: t_std = 0.000001
        else: t_std = np.std(seq)
        seq_n = (seq - np.mean(seq))/t_std
    elif sel == 'zero-mean':
        seq_n = seq - np.mean(seq)
    elif sel == 'euclidean':
        seq_n = seq
    return seq_n

def compute_diff_dist(seq_l, seq_s):
    win_len = len(seq_s)
    dist = []
    if len(seq_l) == len(seq_s):
        d_v = np.array(seq_l)-np.array(seq_s)
        return d_v
    for i in range(len(seq_l)-len(seq_s)):
        t_d = np.array(seq_l[i:i+win_len]) - np.array(seq_s)
        dist.append(np.sum(abs(t_d)))

    idx = np.argmin(dist)
    d_v = np.array(seq_l[idx:idx+win_len]) - np.array(seq_s)
    # return np.min(dist), np.argmin(dist), dist
    return d_v

# def inter_cluster_dist_m(m_subseq, stepsize, slidingWindow):
    # num_cl = len(m_subseq)
    # d_c_mat = np.ones([num_cl, num_cl]) * 10000
    # for i in range(num_cl -1):
        # for j in range(i+1, num_cl):
            # d_k = []
            # for k in range(stepsize):
                # to_test = m_subseq[j][k*slidingWindow:(k+1)*slidingWindow]
                # d_k.append(abs(compute_diff_dist(m_subseq[i], to_test)))
            # d_c_mat[i][j] = d_c_mat[j][i] = np.max(np.sum(d_k, axis=1))
# 
    # return d_c_mat

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
        d_t = [np.linalg.norm(seq) for seq in d_subseq[i]]
        d_ci.append(np.mean(d_t))
        d_c_std.append(np.std(d_t))

    # th_ci = [(np.mean(d)+3*np.std(d))/2 for d in d_ci]

    return np.array(d_ci), np.array(d_c_std) #, th_ci

def get_frequency(W_t, list_idx):
    cnt_f = []
    for i in range(len(list_idx)-1):
        num_cnt = 0
        for j in range(i+1, len(list_idx)):
            if list_idx[j] - list_idx[i] <= W_t: num_cnt+=1
            else:
                if num_cnt >0: cnt_f.append(num_cnt)
                break
    if len(cnt_f) >1:  
        return np.mean(cnt_f)
    else:
        return 3
    
def get_index_diff(list_idx):
    nums = [len(l) for l in list_idx]
    # print('list:', list_idx)
    print('nums:', nums)
    W_t, F_t = np.array([]), np.array([])
    for i in range(len(list_idx)):
        t_list = np.array(list_idx[i])
        t_d = t_list[1:] - t_list[:-1]
        # print('CHK:', i, t_d, t_list)
        # W_t = np.append(W_t, min(round(np.mean(t_d)*2), np.max(t_d)+1))
        if len(t_list) >1:
            # print('chk4', i, t_list, t_d)
            W_t = np.append(W_t, np.min([round((np.mean(t_d)+1)*2), round(np.sum(nums)*0.3)]))
            F_t = np.append(F_t, int(get_frequency(W_t[-1], t_list))+1)
        else:
            W_t = np.append(W_t, round(np.sum(nums)*0.3))
            F_t = np.append(F_t, 3)
    
    ## Need to re-organize W_t, F_t based on weights/statistics of each clusters
    # W_t = W_t*nums/np.min(nums)
    return W_t, F_t #, th_time




###########################################################################
#def compute_min_score(ts, pattern_length, nms, scores_nms, normalize):
#    # Compute score
#    all_join = []
#    for index_name in range(len(nms)):            
#        
#        join = stumpy.stump(ts,pattern_length,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
# 	   #join,_ = mp.join(nm_name + '/' + str(index_name),ts_name,len(nms[index_name]),len(ts), self.pattern_length)
#        join = np.array(join)
#        all_join.append(join)
#
#    join = [0]*len(all_join[0]) # all_join 으로부터 join을 계산. all_join은 각 cluster의 mean subsequnece와 전체 time series의 join값
#    for sub_join,scores_sub_join in zip(all_join,scores_nms):
#        join = [float(j) + float(sub_j)*float(scores_sub_join) for j,sub_j in zip(list(join),list(sub_join))]
#    join = np.array(join)
#
#    # join_n = running_mean(join, pattern_length)
#    # join_n = np.array([join_n[0]]*(pattern_length//2) + list(join_n) + [join_n[-1]]*(pattern_length//2))
#    join_n = join
#    
#    return join_n

###########################################################################
#def chk_other_cluster(seq, cl_NMs, cl_Ws, window_size,  normalize):
#    d_cl = []
#    score_cl = []
#    for cl in range(len(cl_NMs)):
#        t_score = compute_min_score(seq, window_size, cl_NMs[cl], cl_Ws[cl], normalize)
#        d_cl.append(t_score)
#        # score_cl.append(t_score[:window_size])
#
#    return np.min(d_cl), np.argmin(d_cl)

###########################################################################
#def compute_mp(seq, l, nms, normalize):
#    # Compute score
#    all_join = []
#    for index_name in range(len(nms)):            
#        join = stumpy.stump(seq,l,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
#        all_join.append(np.min(join))
#
#    return np.min(all_join), all_join

###########################################################################
#def chk_add_nm(L, thres_cls, normalize, init=False):
#    ## Check how many 'anomalies' in the L
#    to_check, to_idx = L.is_New_Pattern(thres_cls)
#    if len(to_check) ==0 : return []
#
#    ## compute distances between subseq in to_check
#    candidate = to_check[-L.win:]
#    temp_candidate = np.concatenate([candidate, candidate])
#
#    ## TODO: check the unit-length of including candidate (1-by-1 or window_size)
#    while len(to_check) - len(candidate) >= L.win:
#        test = to_check[len(to_check)-len(candidate)-L.win:len(to_check)-len(candidate)]
#        dist = stumpy.stump(test, L.win, temp_candidate, ignore_trivial=False, normalize=normalize, p=1)[0][0]
#        if init:
#            check_score = stumpy.stump(L.seq, L.win, temp_candidate, ignore_trivial=False, normalize=normalize, p=1)[:,0]
#            # print('TA:', temp_candidate)
#            # print(check_score)
#            temp_th = np.mean(check_score) + 3*np.std(check_score)
#            # print('Temp Init. TH:', temp_th)
#            thres_cls[0] = temp_th
#
#        if dist < thres_cls[int(L.idx_cl[-1])]: ## curr. threshold
#            candidate = np.concatenate([test, candidate])
#            temp_candidate = candidate
#        else:
#            break
#    
#    if len(candidate) >= 3*L.win:
#        return candidate[-3*L.win:]
#    else:
#        return []

############################################################################
#def running_mean(x,N):
#	return (np.cumsum(np.insert(x,0,0))[N:] - np.cumsum(np.insert(x,0,0))[:-N])/N
# Revised by JP
#def compute_score(ts, pattern_length, nms, scores_nms, normalize):
#    # Compute score
#    all_join = []
#    for index_name in range(len(nms)):            
#        join = stumpy.stump(ts,pattern_length,nms[index_name],ignore_trivial = False, normalize=normalize, p=1)[:,0]
# 	   #join,_ = mp.join(nm_name + '/' + str(index_name),ts_name,len(nms[index_name]),len(ts), self.pattern_length)
#        join = np.array(join)
#        all_join.append(join)
#
#    join = [0]*len(all_join[0]) # all_join 으로부터 join을 계산. all_join은 각 cluster의 mean subsequnece와 전체 time series의 join값
#    for sub_join,scores_sub_join in zip(all_join,scores_nms):
#        join = [float(j) + float(sub_j)*float(scores_sub_join) for j,sub_j in zip(list(join),list(sub_join))]
#    join = np.array(join)
#    join_n = running_mean(join,pattern_length)
#        # join_n = join
#    
#    #reshifting the score time series
#    join_n = np.array([join_n[0]]*(pattern_length//2) + list(join_n) + [join_n[-1]]*(pattern_length//2))
#    if len(join_n) > len(ts):
#        join_n = join_n[:len(ts)]
#
#    # return join_n, all_join
#    return join_n/pattern_length, all_join

def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None, y_pred=None, th=None):
    grader = metricor()
    
    if np.sum(label) != 0:
        R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True) #
    
        L, fpr, tpr= grader.metric_new(label, score, plot_ROC=True, thres= th)
        precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)

    # max_length = min(len(score),len(data), 20000)
    max_length = len(score)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(20, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
    
    # f3_ax1 = fig3.add_subplot(gs[0, :-1])
    f3_ax1 = fig3.add_subplot(gs[0, :])
    plt.tick_params(labelbottom=False)
   
    plt.plot(data[:max_length],'k')
    if np.any(y_pred):
        plt.plot(y_pred[:max_length], 'c')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
        
    # L = [auc, precision, recall, f, Rrecall, ExistenceReward, 
    #       OverlapReward, Rprecision, Rf, precision_at_k]
    # f3_ax2 = fig3.add_subplot(gs[1, :-1])
    f3_ax2 = fig3.add_subplot(gs[1, :])
    # plt.tick_params(labelbottom=False)
    if np.sum(label) != 0:
        L1 = [ '%.2f' % elem for elem in L]
    plt.plot(score[:max_length])
    if th is None:
        plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    else:
        plt.hlines(th,0,max_length,linestyles='--',color='red')
    plt.ylabel('score')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    #plot the data
    # f3_ax3 = fig3.add_subplot(gs[2, :-1])
    f3_ax3 = fig3.add_subplot(gs[2, :])
    if th is None:
        index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    else:
        index = (label + 2*(score > th))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    
    
    # f3_ax4 = fig3.add_subplot(gs[0, -1])
    # # plt.plot(fpr, tpr)
    # plt.plot(R_fpr,R_tpr)
   ##  plt.title('R_AUC='+str(round(R_AUC,3)))
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
   ##  plt.legend(['ROC','Range-ROC'])
    
    # f3_ax5 = fig3.add_subplot(gs[1, -1])
    # plt.plot(recall, precision)
    # plt.plot(R_tpr[:-1],R_prec)   # I add (1,1) to (TPR, FPR) at the end !!!
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(['PR','Range-PR'])
    if np.sum(label) != 0:
        plt.suptitle(fileName + '    window='+str(slidingWindow) +'   '+ modelName
        +'\nAUC='+L1[0]+'     R_AUC='+str(round(R_AUC,2))+'     Precision='+L1[1]+ '     Recall='+L1[2]+'     F='+L1[3]
        + '     ExistenceReward='+L1[5]+'   OverlapReward='+L1[6]
        +'\nAP='+str(round(AP,2))+'     R_AP='+str(round(R_AP,2))+'     Precision@k='+L1[9]+'     Rprecision='+L1[7] + '     Rrecall='+L1[4] +'    Rf='+L1[8]
        )
    # L = printResult(data, label, score, slidingWindow, fileName=name, modelName=modelName)
    column = ['auc', 'precision', 'recall', 'f', 'Rrecall', 'ExistenceReward', 'OverlapReward', 'Rprecision', 'Rf', 'precision_at_k']

    df = pd.DataFrame(columns=column)
    L[0] = R_AUC
    df = df.append(pd.Series(L, index=df.columns), ignore_index=True)
    print(df)
    return df

def plotFigRev(data, label, scores, slabels, slidingWindow, se, plotRange=None, y_pred=None, th=None, th_addd = None, fname='temp.png'):
    grader = metricor()
    
    # if np.sum(label) != 0:
        # R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True) #
    # 
        # L, fpr, tpr= grader.metric_new(label, score, plot_ROC=True, thres= th)
        # precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)

    # max_length = min(len(score),len(data), 20000)
    max_length = len(label)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(20, 3*len(scores)+4), constrained_layout=True)
    gs = fig3.add_gridspec(len(scores)+1, 4)
    
    # f3_ax1 = fig3.add_subplot(gs[0, :-1])
    f3_ax1 = fig3.add_subplot(gs[0, :])
    plt.tick_params(labelbottom=False)
   
    plt.plot(data,'k', label='Normal-1')
    if len(data) < se[2]:
        plt.plot(range(se[2], se[3]), data[se[2]:se[3]],'b', label='Normal-2')
    if isinstance(se[0], int):        
        if se[0] > 0:
            plt.plot(range(se[0], se[1]), data[se[0]:se[1]], 'k')
    elif len(se[0]) >1:
        print('CHK:', se[0])
        for s1, s2 in zip(se[0], se[1]):
            plt.plot(range(s1, s2), data[s1:s2], 'k')
    if np.any(y_pred):
        plt.plot(y_pred[:max_length], 'c')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r', label='Anomalies')
    # plt.xlim([0,max_length])
    plt.xlim(plotRange)
    plt.legend(bbox_to_anchor=(0.5, 1.4), ncols=3, loc='upper center', facecolor='None')
    
    for i, score in enumerate(scores):
        # L = [auc, precision, recall, f, Rrecall, ExistenceReward, 
        #       OverlapReward, Rprecision, Rf, precision_at_k]
        # f3_ax2 = fig3.add_subplot(gs[1, :-1])
        f3_ax2 = fig3.add_subplot(gs[1+i, :])
        # plt.tick_params(labelbottom=False)
        # if np.sum(label) != 0:
            # L1 = [ '%.2f' % elem for elem in L]
        plt.plot(score[:max_length], label=slabels[i])
        if th is None:
            plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
        else:
            if i == 3:                
                plt.plot(th_addd, 'r:')
            else:
                plt.hlines(th,0,max_length,linestyles='--',color='red')
        plt.ylabel('score')
        # plt.xlim([0,max_length])
        plt.xlim(plotRange)
        # plt.legend(loc='lower left')
        plt.legend(loc='lower right')

    plt.savefig(fname, bbox_inches='tight')
    # plt.savefig('two_normal_local.png', bbox_inches='tight')

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