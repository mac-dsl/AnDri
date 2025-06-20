import numpy as np
from scipy.cluster.hierarchy import distance

from util.util_andri import *
from util.ahc import membership, adaptive_ahc
from util.plot_andri import *

from enum import Enum

class status_window(Enum):
    ACTIVE =1
    INACTIVE =2
    ADD =3

class NormalModel:
    def __init__(self, subseq, tau, nu, c_m, c_std, active=True):
        self.subseq = subseq
        self.tau = tau
        self.nu = nu
        self.m = c_m
        self.std = c_std
        self.active = active

    def set_active(self, status):
        self.active = status

class windowL:
    def __init__(self, W, slidingWindow):
        self.W = W
        self.slidingWindow = slidingWindow
        self.NM = []
        self.mem = []
        self.seq = []
        self.dist = []
        self.avg_score = []
        self.prepare_add = False
        # self.cl = []

    def update_NM(self, NormalModel):
        self.NM.append(NormalModel)
        self.mem.append([])
        self.dist.append([])
        self.avg_score.append([])
        if len(self.seq) > 0:
            for se in self.seq:
                m_t, d_t = membership(NormalModel.subseq, se, NormalModel.tau, eta=1)
                self.mem[-1] = np.append(self.mem[-1], m_t)
                self.dist[-1] = np.append(self.dist[-1], d_t)
                self.avg_score[-1] = np.append(self.avg_score[-1], np.mean(self.dist[-1]))

    def enqueue(self, seq, idx):
        self.seq.append(seq)
        num_active = len([nm for nm in self.NM if nm.active == True])
        return_status = []

        ## computing membership function for all normal models and change activeness
        nm_turn_act, nm_turn_inact = [], []
        for i, nm in enumerate(self.NM):
            m_t, d_t = membership(nm.subseq, seq, nm.tau, eta=1)
            self.mem[i] = np.append(self.mem[i], m_t)
            self.dist[i] = np.append(self.dist[i], d_t)
            self.avg_score[i] = np.append(self.avg_score[i], np.mean(self.dist[i]))

            ## Queuing W
            if len(self.seq) > self.W:
                self.mem[i] = np.delete(self.mem[i], 0)
                self.dist[i] = np.delete(self.dist[i], 0)
                self.avg_score[i] = np.delete(self.avg_score[i], 0)

                ## When checking stats, exclude the minimum membership in W
                curr_mem = copy.deepcopy(self.mem[i])
                if self.prepare_add: curr_mem = curr_mem[-(self.W//2):]
                curr_mem = np.delete(curr_mem, np.argmin(curr_mem))

                ## Active --> Inactive based on membership
                if nm.active == True and np.mean(curr_mem) < nm.nu:
                    print(f'[<-- Inactive]: NM {i} = {np.mean(curr_mem)}/{nm.nu}, at {idx}')
                    nm.set_active(False)
                    num_active -=1
                    return_status.append(status_window.INACTIVE)
                    nm_turn_inact.append(i)

                ## Inactive --> Active based on membership
                elif nm.active == False and np.mean(curr_mem) >= nm.nu:
                    print(f'[--> Active]: NM {i} = {np.mean(curr_mem)}/{nm.nu}, at {idx}')
                    nm.set_active(True)
                    num_active +=1
                    return_status.append(status_window.ACTIVE)
                    nm_turn_act.append(i)

        if len(self.seq) > self.W: 
            if self.prepare_add:
                return_status.append(status_window.ADD)
            else:
                ## similar but small std cases
                nm_idx = np.argmax([np.mean(self.mem[i]) for i in range(len(self.NM))])
                nm = self.NM[nm_idx]
                if len(self.seq) > self.W and nm.active ==True and np.mean(self.mem[nm_idx]) > nm.nu:
                    if nm.tau < np.mean(self.dist[nm_idx])+3*np.std(self.dist[nm_idx]) and nm.std> np.std(self.dist[nm_idx]):
                        print(f'[{idx}], NM {i}, tau: {nm.tau:.2f} (M: {nm.m}, STD: {nm.std}) vs. {np.mean(self.dist[nm_idx])+3*np.mean(self.dist[nm_idx])}, (M: {np.mean(self.dist[nm_idx])}, STD: {np.mean(self.dist[nm_idx])})')
                        return_status.append(status_window.INACTIVE)
            
            del self.seq[0]

        if num_active <=0:
            nm_f_idx =self.force_active()
            num_active += 1
            return_status.append(status_window.ACTIVE)
            nm_turn_act.append(nm_f_idx)

        return return_status, nm_turn_act, nm_turn_inact
        
    def terminate_add(self, org_W):
        if self.prepare_add: self.prepare_add = False
        if self.W > org_W:
            for j in range(self.W - org_W):
                for i, nm in enumerate(self.NM):
                    self.mem[i] = np.delete(self.mem[i], 0)
                    self.dist[i] = np.delete(self.dist[i], 0)
                    self.avg_score[i] = np.delete(self.avg_score[i], 0)
                del self.seq[0]
                
        self.W = org_W


    def force_active(self):
        nm_mems = [np.mean(mem) for mem in self.mem]
        self.NM[int(np.argmax(nm_mems))].set_active(True)
        return int(np.argmax(nm_mems))


class AnDri():

    def __init__(self,pattern_length, normalize='zero-mean', linkage_method='ward', th_reverse=5, 
                 kadj=1, nm_len=2, overlap=0.5, max_W = 15, eta=1, REVISE_SCORE=True, device_id=0):
        self.pattern_length = pattern_length
        self.normalize=normalize     
        self.overlap = overlap
        self.linkage_method=linkage_method
        self.th_reverse=th_reverse
        self.kadj = kadj
        self.nm_len = nm_len
        self.seq_idx = 0
        self.eta = eta
        self.Z = []
        self.REVISE_SCORE = REVISE_SCORE
        # self.div_seq = div_seq

        self.scores = np.array([])
        self.cl_s = np.array([])
        self.est_label = np.array([])
        self.max_W = max_W
        self.W = max_W

        self.listcluster = []
        self.cl_list = []
        self.mem_nm = []
        self.dist_nm = []
        self.device_id = device_id
    

    def __training(self, training_len, stepwise, min_size=0.025):
        """
        Training using Adjacent Hierarchical Clustering
        """
        x_train =  self.ts[:training_len]
        if self.y is not None: y_train = self.y[:training_len]
        
        ## Divide training set w/ subseq. length
        seqs_n = []
        if self.y is not None: n_subseq, n_label = divide_subseq(x_train, self.pattern_length, self.nm_len, overlap=self.overlap, label=y_train)
        else: n_subseq = divide_subseq(x_train, self.pattern_length, self.nm_len, overlap=self.overlap)

        ## normalize subseqences
        for seq in n_subseq: seqs_n.append(norm_seq(seq, self.normalize))
        seqs_n = np.array(seqs_n)
        
        ## Adjacent Hierarchical Clustering TODO: zero-mean vs. SBD
        listcluster, Z, m_subseq, m_tau, m_nu, _, __cached__, d_ci, d_c_std, Ws, _ = adaptive_ahc(
            seqs_n, linkage_method=self.linkage_method, th_reverse=self.th_reverse, kadj=self.kadj, eta=self.eta, max_W = self.max_W, min_size =min_size)

        if Ws is None: return None
        self.Z = Z
        
        self.W = int(np.min([self.max_W, 2*np.max(Ws)]))
        for i in range(len(m_tau)):
            print(f'NM: {i}, Tau: {m_tau[i]}, Nu: {m_nu[i]}, M: {d_ci[i]}, STD: {d_c_std[i]}')

        ## Normal Models
        self.NMs = []
        for m_s, m_t, m_n, m_m, m_std in zip(m_subseq, m_tau, m_nu, d_ci, d_c_std): self.NMs.append(NormalModel(m_s, m_t, m_n, m_m, m_std))
        for i in range(len(self.NMs)): 
            self.mem_nm.append([])
            self.dist_nm.append([])

        self.listcluster = listcluster
        cl_list = np.array([])
        for lc in listcluster: cl_list = np.append(cl_list, np.ones(int(self.pattern_length*self.nm_len*(1-self.overlap)))*lc)

        if len(cl_list) < len(x_train): cl_list = np.append(cl_list, np.ones(len(x_train)-len(cl_list))*(-1))
        self.cl_s = cl_list
        
        self.cl_s = []
        before_status = False
        for i in range(0, len(x_train), self.pattern_length):
            x_test = x_train[i:i+self.pattern_length]
            x_test_n = norm_seq(x_test, self.normalize)
            for j, nm in enumerate(self.NMs): 
                m_j, d_j = membership(nm.subseq, x_test_n, nm.tau, eta=1)
                self.mem_nm[j] = np.append(self.mem_nm[j], m_j)
                self.dist_nm[j] = np.append(self.dist_nm[j], d_j)

            score_cl, _ = [], []
            ## For cluster members
            if cl_list[i] != -1:
                nm_idx = int(cl_list[i])
                score = np.linalg.norm(compute_diff_dist(self.NMs[nm_idx].subseq, x_test_n))
                self.cl_s = np.append(self.cl_s, np.ones(self.pattern_length) * nm_idx)                

            else:
                for j, nm in enumerate(self.NMs):
                    score_cl.append(np.linalg.norm(compute_diff_dist(nm.subseq, x_test_n)))

                score = np.min(score_cl)
                self.cl_s = np.append(self.cl_s, np.ones(self.pattern_length) * np.argmin(score_cl))
                nm_idx = int(np.argmin(score_cl))

            selected_nm = self.NMs[nm_idx]
            selected_th = selected_nm.tau

            if stepwise:
                if i >= self.pattern_length: x_tests = x_train[i-self.pattern_length:i+self.pattern_length]
                else: x_tests = x_train[i:i+self.pattern_length*2]
                rev_score = backward_anomaly2(x_tests, self.pattern_length, self.NMs[nm_idx].subseq, self.normalize, self.device_id)
                if i >= self.pattern_length: self.scores = np.append(self.scores, rev_score[:self.pattern_length])
                else: self.scores = np.append(self.scores, rev_score[:self.pattern_length//2])
            else:
                ## Find sub-subseq. of anomalies
                if score > selected_th:
                    tmp_score = backward_anomaly(x_train[i-self.pattern_length:i+self.pattern_length], self.pattern_length, self.NMs[nm_idx].subseq, self.normalize)
                    tmp_score = tmp_score/self.NMs[nm_idx].tau
                    self.scores = np.append(self.scores, tmp_score)
                    before_status = True
                else:
                    if before_status:
                        tmp_score = backward_anomaly(x_train[i-self.pattern_length:i+self.pattern_length], self.pattern_length, self.NMs[nm_idx].subseq, self.normalize)
                        tmp_score = tmp_score/self.NMs[nm_idx].tau
                        self.scores = np.append(self.scores, tmp_score)
                    else:
                        self.scores = np.append(self.scores, np.ones(self.pattern_length)*score/self.NMs[nm_idx].tau)
                    before_status = False

    #  @params X: Time-series
    #  @params y: labels (not necessary)
    #  @params online: True (online), False (offline)
    #  @params delta: default =0, W for best case (to revise scores)
    #  @params training_len: training len in ratio (ex. 0.2 for 20%)
    #  @params stepwise: default FALSE, for comparison only
    #  @params align: default True (for aligning scores for multiple normal patterns)
    #  @params min_size: ratio of minimum size of cluster (R_{min})
    def fit(self, X, y=None, online=True, delta=0, training_len=None, stepwise=False, align=True, min_size=0.025):
        ## if overlapping is not 1, revise for loop 
        self.ts = X
        self.y = y

        if online:
            if training_len < len(X)*0.1:
                print('[ERR]: Need to specify training length (>= 10% of data)')
                return 0
            
            self.__training(training_len, stepwise, min_size)
            # print('Train:', training_len, 'vs', len(self.scores), 'start:', training_len)
            if len(self.scores) ==0: return None

            movingWin = windowL(self.W, self.pattern_length)
            ## Normal models are already saved.
            for nm in self.NMs: movingWin.update_NM(nm)

            print(f'START: {len(self.mem_nm)}, {len(self.mem_nm[0])}')

            ## Compute score for each subsequence 
            self.seq_idx = training_len
            before_status = False

            while (self.seq_idx < len(X)):
                test_seq = self.ts[self.seq_idx:self.seq_idx+self.pattern_length]
                test_seq_n = norm_seq(test_seq, self.normalize)
                if len(test_seq) < self.pattern_length:
                    if stepwise:
                        x_tests = self.ts[self.seq_idx-self.pattern_length:]
                        if type(selected_nm) == list: selected_nm = selected_nm[0]
                        rev_score = backward_anomaly2(x_tests, self.pattern_length, selected_nm.subseq, self.normalize, self.device_id)
                        self.scores = np.append(self.scores, rev_score[:self.pattern_length])
                    print(f'Online scores len: {len(self.scores)}')
                    print('Done')
                    break

                rsts, nm_acts, _ = movingWin.enqueue(test_seq_n, self.seq_idx)


                if status_window.ACTIVE in rsts:
                    if self.REVISE_SCORE: self.revise_scores(movingWin, delta, nm_acts, stepwise, status_window.ACTIVE)

                elif status_window.INACTIVE in rsts:
                    if movingWin.W == self.W:
                        movingWin.W = self.W*2
                        movingWin.prepare_add = True

                if status_window.ADD in rsts:
                    new_NM, d_nm, new_len = self.add_new_NM(self.ts[self.seq_idx-movingWin.W*self.pattern_length:self.seq_idx+self.pattern_length], self.nm_len, self.overlap, movingWin.NM, self.seq_idx)
                    if new_NM is not None:
                        if new_NM.tau > max([nm.tau for nm in self.NMs]): new_NM.tau = max([nm.tau for nm in self.NMs])
                        if new_NM.nu > max([nm.nu for nm in self.NMs]): new_NM.nu = max([nm.nu for nm in self.NMs])
                        self.NMs.append(new_NM)
                        movingWin.update_NM(new_NM)
                        self.mem_nm.append([])
                        self.dist_nm.append([])
                        print(f'[==> Add NM]: NM# {len(self.NMs)-1}, TAU: {new_NM.tau}, Nu: {new_NM.nu}, LEN: {new_len}')

                        if self.REVISE_SCORE: self.revise_scores(movingWin, delta, len(self.NMs)-1, stepwise, status_window.ADD)

                    movingWin.terminate_add(self.W) 
                
                for j in range(len(movingWin.NM)):
                    self.mem_nm[j] = np.append(self.mem_nm[j], movingWin.mem[j][-1])
                    self.dist_nm[j] = np.append(self.dist_nm[j], movingWin.dist[j][-1])

                ## Anomaly score with active normal patterns
                score_cl, i_cl, score_tau = [],[],[]
                for j, nm in enumerate(movingWin.NM):
                    if nm.active:
                        score_cl.append(movingWin.dist[j][-1])
                        i_cl.append(j)

                ## save current cluster, score
                score, nm_idx = np.min(score_cl), i_cl[np.argmin(score_cl)]
                self.cl_s = np.append(self.cl_s, nm_idx*np.ones(self.pattern_length))

                if stepwise:
                    if self.seq_idx >= self.pattern_length:
                        x_tests = self.ts[self.seq_idx-self.pattern_length:self.seq_idx+self.pattern_length]
                    else: continue

                    ## Find sub-subseq. of anomalies
                    if self.cl_s[-1] == self.cl_s[-self.pattern_length-1]:
                        selected_nm = movingWin.NM[nm_idx]
                        rev_score = backward_anomaly2(x_tests, self.pattern_length, selected_nm.subseq, self.normalize, device_id=self.device_id)
                    else:
                        selected_nm = movingWin.NM[nm_idx]
                        rev_score = backward_anomaly2(x_tests, self.pattern_length, selected_nm.subseq, self.normalize, device_id=self.device_id)
                    
                    self.scores = np.append(self.scores, rev_score[:self.pattern_length])
                else:
                    selected_th = selected_nm.tau
                    if score > selected_th:
                        tmp_score = backward_anomaly(self.ts[self.seq_idx-self.pattern_length:self.seq_idx+self.pattern_length], self.pattern_length, selected_nm.subseq, self.normalize)
                        tmp_score = tmp_score/selected_th
                        self.scores = np.append(self.scores, tmp_score)
                        before_status = True
                    else:
                        if before_status:
                            tmp_score = backward_anomaly(self.ts[self.seq_idx-self.pattern_length:self.seq_idx+self.pattern_length], self.pattern_length, selected_nm.subseq, self.normalize)
                            tmp_score = tmp_score/selected_th
                            self.scores = np.append(self.scores, tmp_score)
                        else:
                            self.scores = np.append(self.scores, np.ones(self.pattern_length)*score/movingWin.NM[nm_idx].tau)
                        before_status=False
 
                self.seq_idx += self.pattern_length

        ## Offline method
        else:
            self.__training(len(X), stepwise, min_size)
            print(f'Offline scores len: {len(self.scores)}')
            if len(self.scores) == 0:
                return None

        self.scores = np.append(self.scores, self.scores[-1]*np.ones(len(X)-len(self.scores)))
        if align:
            self.scores = align_score(self.NMs, self.scores, self.cl_s[:len(self.scores)], self.pattern_length)
        else:
            self.scores = running_mean(self.scores, self.pattern_length)
            self.scores = np.array([self.scores[0]]*((self.pattern_length-1)//2) + list(self.scores) + [self.scores[-1]]*((self.pattern_length-1)//2))
    
    ## Update scores and cl_s from curr_idx
    def update_score(self, movingWin, curr_idx, nm_idx, score, stepwise, before_status):
        self.cl_s[curr_idx:curr_idx+self.pattern_length] = np.ones(self.pattern_length)*nm_idx
        x_tests = self.ts[curr_idx-self.pattern_length:curr_idx+self.pattern_length]
        
        if stepwise:
            if self.cl_s[curr_idx+1] == self.cl_s[curr_idx-self.pattern_length+1]:
                selected_nm = movingWin.NM[nm_idx]
                rev_score = backward_anomaly2(x_tests, self.pattern_length, selected_nm.subseq, self.normalize, device_id=self.device_id)
            else:
                selected_nm = []
                selected_nm.append(movingWin.NM[int(self.cl_s[curr_idx-self.pattern_length+1])])
                selected_nm.append(movingWin.NM[nm_idx])
                rev_score = backward_anomaly_changing_point(x_tests, self.pattern_length, selected_nm, self.normalize, device_id=self.device_id)

            
        else:
            selected_nm = movingWin.NM[nm_idx]
            selected_th = selected_nm.tau
            if score > selected_th:
                tmp_score = backward_anomaly(x_tests, self.pattern_length, selected_nm.subseq, self.normalize)
                tmp_score = tmp_score/selected_th
                self.scores = np.append(self.scores, tmp_score)
                before_status = True
            else:
                if before_status:
                    tmp_score = backward_anomaly(x_tests, self.pattern_length, selected_nm.subseq, self.normalize)
                    rev_score = tmp_score/selected_th
                else:
                    rev_score = np.ones(self.pattern_length)*score/movingWin.NM[nm_idx].tau
                before_status=False
                
        return rev_score, before_status


    ## After adding a new Normal Model, revise scores, cl_s, and memberships and distance for each NMs
    def revise_scores(self, movingWin, delta, nm_ids, stepwise, added=status_window.ADD, before_status=False):
        # rev_cl_s,rev_scores = np.array([]), np.array([])
        before_status = False
        if delta > movingWin.W*self.pattern_length: delta = movingWin.W*self.pattern_length
        start_idx = self.seq_idx - (delta-delta%self.pattern_length)
        
        while (start_idx < self.seq_idx):
            curr_cl = int(self.cl_s[start_idx+1])
            win_idx = int((self.seq_idx - start_idx)/self.pattern_length) -1
            min_dist, min_j = movingWin.dist[curr_cl][-win_idx], curr_cl

            ## When active...
            if added == status_window.ACTIVE:
                ## Compare movingWin.dist & movingWin.mem
                for j in nm_ids:
                    if min_dist > movingWin.dist[j][-win_idx]:
                        min_dist, min_j = movingWin.dist[j][-win_idx], j
                if min_j == curr_cl: 
                    start_idx += self.pattern_length
                    continue
                else:
                    ## update cl_s and scores
                    rev_score, before_status = self.update_score(movingWin, start_idx, min_j, min_dist, stepwise, before_status)
                    self.scores[start_idx-self.pattern_length//2-1:start_idx-self.pattern_length//2-1 + self.pattern_length] = rev_score[:self.pattern_length]

            elif added == status_window.ADD:
                new_dist = np.linalg.norm(compute_diff_dist(movingWin.NM[nm_ids].subseq, self.ts[start_idx:start_idx+self.pattern_length]))
                if min_dist > new_dist:
                    min_dist, min_j = new_dist, nm_ids
                if min_j == curr_cl: 
                    start_idx += self.pattern_length
                    continue
                else:
                    rev_score, before_status = self.update_score(movingWin, start_idx, min_j, min_dist, stepwise, before_status)
                    self.scores[start_idx-self.pattern_length//2-1:start_idx-self.pattern_length//2-1 + self.pattern_length] = rev_score[:self.pattern_length]

                m_t, d_t = membership(movingWin.NM[nm_ids].subseq, self.ts[start_idx:start_idx+self.pattern_length], movingWin.NM[nm_ids].tau, eta=1)
                self.mem_nm[-1] = np.append(self.mem_nm[nm_ids], m_t) 
                self.dist_nm[-1] = np.append(self.dist_nm[nm_ids], d_t)
            else:
                return

            start_idx += self.pattern_length


    #  @params seqs: divided subsequences
    #  @params step: step for dividing subsequences
    #  @params overlap: overlap of dividing (i.e., 0: no overlap)
    #  @params NMs: current Normal Model
    def add_new_NM(self, t_seq, step, overlap, NMs, idx):
        """
        Try to add new normal model online 
        """

        curr_len, chk_len = self.pattern_length, find_length(t_seq[::-1])
        chk_len = curr_len

        ## divide sequences with the new chk_len
        win_seqs = divide_subseq(t_seq, chk_len, step, 0)
        win_seqs_n = []
        for seq in win_seqs: win_seqs_n.append(norm_seq(seq, self.normalize))
        win_seqs_n = np.array(win_seqs_n)
        listcluster, Z, m_subseq, m_tau, m_nu, _, _, d_ci, d_c_std, _, _ = adaptive_ahc(
            win_seqs_n, linkage_method='ward', th_reverse=10, kadj=1, eta=self.eta, max_W = self.max_W, isTrain=False, NMs=NMs)

        if listcluster is None: return None, None, None

        ## number of members for each cluster
        nums, new_NM = [], []
        for i in range(int(max(listcluster)+1)): nums.append(len([k for k in listcluster if k==i]))

        j = 0
        for m_s, m_tau, m_nu, m_m, m_std in zip(m_subseq, m_tau, m_nu, d_ci, d_c_std):
            chk = True
            for nm_i, nm in enumerate(NMs):
                ## To resolve align error
                temp_dist1 = np.linalg.norm(compute_diff_dist(nm.subseq, m_s[:len(nm.subseq)//2]))
                temp_dist2 = np.linalg.norm(compute_diff_dist(nm.subseq, m_s[len(nm.subseq)//2:]))
                temp_dist = (temp_dist1+temp_dist2) 

                ## If new m_subseq. is similar to current NM, update tau instead of adding it
                if temp_dist <= nm.tau:
                    ## Update selected NM (status) and tau
                    if nm.active == False:
                        nm.set_active(True)
                        print(f'[--> Active2]: NM {nm_i} from exam. at {idx}')
                        nm.nu = (nm.nu + m_nu)/2

                    nums[j] = 0
                    chk = False
                    break
                
            if m_nu < np.min([nm.nu for nm in NMs]):
                chk = False
                nums[j] = 0
            elif chk:
                ## add candidate NM
                new_NM.append(NormalModel(m_s, m_tau, m_nu, m_m, m_std, active=True))
            j +=1

        for j in range(len(nums)-1, -1, -1): 
            if nums[j] == 0: del nums[j]

        if len(new_NM) >0:

            new_n = []
            for j, nm in enumerate(new_NM):
                new_n.append(nm.subseq)

            d_m = distance.squareform(distance.pdist(new_n))
            if np.max(nums) == 0:
                return None, None, None
            else:
                return new_NM[np.argmax(m_nu)], d_m, chk_len
        else:
            return None, None, None

