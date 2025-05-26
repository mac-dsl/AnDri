import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, distance
from util.util_a2d2 import intra_cluster_dist, compute_diff_dist, longest_consecutive_sequence
# from util.util_overlap import intra_cluster_dist, compute_diff_dist, longest_consecutive_sequence
# import logging
import copy
# from numpy.linalg import norm, eigh
# from numpy.fft import fft, ifft
import math
# import seaborn as sns
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray', 
          'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan', 'pink', 'deepskyblue', 'lawngreen',
          'royalblue', 'darkgrey', 'darkorange', 'darkgreen','darkviolet','salmon','olivedrab','lightcoral','darkcyan','yellowgreen']
markers = ['o', 'x', '^', 'v', 's', '*', '+', '.', ',', '<', '>' , '1','2','3','4','p','h','H','D','d']

MAX_DIST = 100000

#  @params seqAs, seqBs: target subseqeuences (in each cluster) to compute distance
#  @params: linkage_method: linkage method = {'average', 'complete', 'single', 'centroid', 'ward}
def __compute_dist_cls(seqAs, seqBs, linkage_method='average'):
    """
    Compute linkage between clusters
    """
    if linkage_method in ['average', 'complete', 'single']:
        d_AB = []
        for seqA in seqAs:
            for seqB in seqBs:
                d_AB.append(np.linalg.norm(compute_diff_dist(np.array(seqA), np.array(seqB))))
        if linkage_method =='average': d_res = np.mean(d_AB)
        elif linkage_method == 'complete': d_res = np.max(d_AB)
        elif linkage_method == 'single': d_res = np.min(d_AB)
    elif linkage_method == 'centroid': d_res = np.linalg.norm(compute_diff_dist(np.mean(seqAs, axis=0), np.mean(seqBs, axis=0)))
    elif linkage_method == 'ward':
        m_A = np.mean(seqAs, axis=0)
        m_B = np.mean(seqBs, axis=0)
        d_res = np.linalg.norm(m_A-m_B)**2 * (len(seqAs)*len(seqBs)/(len(seqAs)+len(seqBs)))
    else:
        print('Linkage:', linkage_method, 'is not provided.')
        return None
    return d_res

def __adj_dist(seqs, linkage_method):
    d_vec = []
    for i in range(len(seqs)-1):
        d_vec = np.append(d_vec, __compute_dist_cls([seqs[i]], [seqs[i+1]], linkage_method=linkage_method))
    return d_vec

#  @params d_mat: pairwise distance matrix (n-by-n)
#  @params kadj: k-adjacent neighbors
#  @params cl_s: set of clusters
def __find_min(d_mat, kadj, cl_s):
    """
    Find the closest clusters within range k-adj
    """
    if kadj == 1:
        ind, min_d = np.argmin(d_mat), np.min(d_mat)
        tid1 = int(ind/len(d_mat))
        tid2 = ind -tid1*len(d_mat)
        if tid1 < tid2: left, right = tid1, tid2
        else: left, right = tid2, tid1
    
    else:
        d_mat_sub = []
        if len(d_mat) > kadj:
            for i in range(len(d_mat)-kadj):
                d_mat_sub.append(d_mat[i, i+1:i+1+kadj])
            ind, min_d = np.argmin(d_mat_sub), np.min(d_mat_sub)
            tid1 = int(ind/kadj)
            tid2 = ind - tid1*kadj
            left, right = tid1, tid1+tid2+1
        else:
            ind, min_d = np.argmin(d_mat), np.min(d_mat)
            tid1 = int(ind/len(d_mat))
            tid2 = ind - tid1*len(d_mat)
            if tid1 < tid2: left, right = tid1, tid2
            else: left, right = tid2, tid1
        

        # d_mat_ij = d_mat.copy()
        # ## Find the minimum distance within d_mat
        # ind, min_d = np.argmin(d_mat_ij), np.min(d_mat_ij)  
# 
        # ## Set a pair with left and right
        # tmp1 = int(ind/len(d_mat_ij))
        # tmp2 = ind - tmp1*len(d_mat_ij)
        # if tmp1 < tmp2: left, right = tmp1, tmp2
        # else: left, right = tmp2, tmp1
# 
        # ## Iteratively find the min of d_mat within range k-adj
        # while np.min(cl_s[right]) - np.max(cl_s[left]) > kadj:
        #     d_mat_ij[tmp1, tmp2] = 100000
        #     ind, min_d = np.argmin(d_mat_ij), np.min(d_mat_ij)  
        #     tmp1 = int(ind/len(d_mat_ij))
        #     tmp2 = ind - tmp1*len(d_mat_ij)
        #     if tmp1 < tmp2: left, right = tmp1, tmp2
        #     else: left, right = tmp2, tmp1

    return left, right, min_d

#  @params left, right: index of clusters to merge 
#  @params cl_s: Clusters containing members' indices
#  @params d_mat: triangular distance matrix
#  @params seqs: subsequences in the training
def __merge_cls(left, right, cl_s, d_mat, seqs, kadj, linkage_method ='average'):
    """
    Update distance matrix d_mat, merge clusters
    """

    cluster_new = copy.deepcopy(cl_s)
    cluster_new[left] += cluster_new[right]
    # del cluster_new[right]

    if kadj==1:
        ## TODO: compute distance of 'new neighbor' after flipped-merge (0403)
        # if left != 0:
            # d_mat[left-1] = __compute_dist_cls([seqs[cl] for cl in cluster_new[left]], [seqs[cl] for cl in cluster_new[left-1]], linkage_method)
        # if left+1 < len(cluster_new):
            # d_mat[left] = __compute_dist_cls([seqs[cl] for cl in cluster_new[left]], [seqs[cl] for cl in cluster_new[left+1]], linkage_method)
# 
        # d_mat = np.delete(d_mat, right-1)
        for i in range(len(d_mat)):
            if i !=left and i!= right:
                # if i == len(d_mat)-1: continue
                if (__min_dist_cl(cluster_new[left], cluster_new[i])) == 1:
                    # print(f'REV: {i}, {cluster_new[i]}')
                    d_mat[left, i] = d_mat[i, left] = __compute_dist_cls([seqs[cl] for cl in cluster_new[left]], [seqs[cl] for cl in cluster_new[i]], linkage_method)

        del cluster_new[right]
        d_mat = np.delete(d_mat, right, axis=0)
        d_mat = np.delete(d_mat, right, axis=1)

    else:
        
        for i in range(len(d_mat)):
            if i !=left and i != right:
                # if i == len(d_mat)-1 : continue
                d_mat[left,i] = d_mat[i,left] = __compute_dist_cls([seqs[cl] for cl in cluster_new[left]], [seqs[cl] for cl in cluster_new[i]], linkage_method)
        
        del cluster_new[right]
        d_mat = np.delete(d_mat, right, axis=0)
        d_mat = np.delete(d_mat, right, axis=1)

    return cluster_new, d_mat

#  @params hist_cluster: history of prev_ clusters (for each steps to roll back)
#  @params target: target idx of Z to split
def __get_sub_lr(hist_cluster, target):
    """
    Split target cluster (roll-back to target status) and return left and right
    """
    if len(target) ==2:
        return [target[0]], [target[1]]
    f_idx = [i for i, part_l in enumerate(hist_cluster) if set(part_l).intersection(set(target)) == set(part_l)]
    c2 = hist_cluster[f_idx[-2]]
    c1 = [x for x in target if x not in c2]
    # print('C12: ', c1, c2)
    
    if np.min(c1) < np.min(c2):
        left, right = c1, c2
    else:
        left, right = c2, c1

    return left, right

#  @params Z: matrix Z of hierarchical clustering 
#  @params add_pair: history of added clusters for each step
#  @params clusters: current sub-clusters
#  @params left, right: selected idx of clusters for merging
#  @params num_seq: num_seq at the start of clustering (to chk Z idx)
def __get_index_Z(Z, add_pair, clusters, left, right, num_seq):
    """
    Find candidate clusters' indices (c1, c2) and previous linkage distances (d_c1, d_c2)
    """
    c1 = num_seq + add_pair.index(clusters[left]) if clusters[left] in add_pair else clusters[left][0] ## left 
    c2 = num_seq + add_pair.index(clusters[right]) if clusters[right] in add_pair else clusters[right][0]  ## right

    d_c1 = Z[c1-num_seq][2] if c1 > num_seq else -1
    d_c2 = Z[c2-num_seq][2] if c2 > num_seq else -1

    return c1, c2, d_c1, d_c2

#  @params cl_a, cl_b: list of clusters when merging
def __min_dist_cl(cl_a, cl_b):
    """
    Find the 'distance' between two sub-clusters (distance: min. distance)
    """
    # min_ab = len(cl_a) + len(cl_b)
    min_ab = np.min(cl_a) + np.max(cl_b)
    for ai in cl_a:
        for bi in cl_b:
            if abs(ai-bi) < min_ab:
                min_ab = abs(ai-bi)

    return min_ab

#  @params seqs: divided sequences
#  @params left, right: selected idx of clusters for merging
#  @params c1, c2: Z-idx of merging 
#  @params c3: distance between c1 and c2
#  @params d_c1, d_c2: inner distance (when merging) of c1 and c2 (when they merged)
#  @params Z: matrix Z of hierarchical clustering 
#  @params d_mat: distance matrix between seqs
#  @params clusters: current sub-clusters
#  @params add_pair: history of added clusters for each step
#  @params num_seq: num_seq at the start of clustering (to chk Z idx)
#  @params th_reverse: % of difference to check flip
#  @params kadj: k-adjacent distance to allow merging
def flipped_merge(seqs, left, right, c1, c2, c3, d_c1, d_c2, Z, d_mat, clusters, add_pair, num_seq, th_reverse, kadj, max_W, linkage_method):
    """
    Do (cascade) flipped merge process (using simple ward and reduce cascade flip)
    """
    # print(f'[START] d_c1: {d_c1}, d_c2: {d_c2}, c3: {c3}')
    while (d_c1-c3 > d_c1*th_reverse/100 or d_c2-c3> d_c2*th_reverse/100):
        t_dist_org = __min_dist_cl(clusters[left], clusters[right])
        
        if kadj >1 and t_dist_org > kadj: 
            # print('[ERR3]:', kadj, '<', t_dist_org)
            break
        # if t_dist_org > max_W:
            # print('[ERR4]:', kadj, '>', t_dist_org)
            # break
        # print(f"FLIPPED: Left: {left} => {clusters[left]}-> {d_c1}, Right: {right} => {clusters[right]}-> {d_c2} and c3: {c3}")
        ## Find later merged cluster and break it 
        if d_c1 > d_c2:
            res, brk_Z_idx, res_idx, brk_idx = c2, c1, right, left
        else:
            res, brk_Z_idx, res_idx, brk_idx = c1, c2, left, right
        
        prev_Z_idx = brk_Z_idx-num_seq

        res_cls = clusters[res_idx].copy()  ## remained cluster        
        brk_l_cls, brk_r_cls = __get_sub_lr(add_pair, add_pair[prev_Z_idx]) ## divide break_cluster

        ## Compute distances from remain-to-divided clusters
        d_res_L =__compute_dist_cls([seqs[id] for id in res_cls], [seqs[id] for id in brk_l_cls], linkage_method)
        d_res_R =__compute_dist_cls([seqs[id] for id in res_cls], [seqs[id] for id in brk_r_cls], linkage_method)
        # print('DLR:', d_res_L, d_res_R, c3, np.mean(np.array(Z)[:,2]))

        if np.min([d_res_L, d_res_R]) > c3 or np.mean(np.array(Z)[:,2]) > c3:   ## prevent cascade flip 
            # print('[ERR2] Stop cascade: ', d_res_L, d_res_R, c3)
            break

        brk_Z_idx = Z[prev_Z_idx][0] if d_res_L < d_res_R else Z[prev_Z_idx][1] ## Z_idx to go back
        # print('[Z IDX] ', prev_Z_idx, brk_Z_idx, '[ADD_PAIR]', len(add_pair), add_pair[prev_Z_idx:])
        # print(f'[ID Findings] p_Z: {prev_Z_idx} with {Z[prev_Z_idx]}. And c12ID: {c1, c2}. BRK_LR: {brk_l_cls, brk_r_cls}')
        

        if brk_Z_idx < num_seq:
            brk_cls, d_brk = [brk_Z_idx], -1    ## for singleton 
        else:
            brk_cls = brk_l_cls if brk_Z_idx == Z[prev_Z_idx][0] else brk_r_cls
            d_brk = Z[brk_Z_idx-num_seq][2] if brk_Z_idx > num_seq else -1
            # d_brk = Z[prev_Z_idx][2]    ## temporal (deleted cluster, distance at that time)
        # print(f'brk_Z_idx: {brk_Z_idx} vs. Prev_Z: {prev_Z_idx+num_seq} LR: {d_res_L} and {d_res_R}')
        
        t_dist = __min_dist_cl(res_cls, brk_cls)    ## find the shorter distance

        if t_dist <= t_dist_org: 
            # print(f'[ERR1]: {res_cls, brk_cls} < {t_dist} or {t_dist_org}')
            break

        ## Revise clusters & d_mat
        del clusters[brk_idx]   ## delete the cluster to break
        if kadj ==1:
            ## TODO: compute distance of 'new neighbor' after flipped-merge (0403)
            # if brk_idx !=0:
                # d_mat[brk_idx-1] = __compute_dist_cls([seqs[id] for id in clusters[brk_idx-1]], [seqs[id] for id in brk_l_cls], linkage_method)
            # if brk_idx+1 < len(clusters):
                # d_mat[brk_idx] = __compute_dist_cls([seqs[id] for id in clusters[brk_idx]], [seqs[id] for id in brk_r_cls], linkage_method)
            # d_mat = np.insert(d_mat, brk_idx, __compute_dist_cls([seqs[id] for id in brk_r_cls], [seqs[id] for id in brk_l_cls], linkage_method))
            # clusters.insert(brk_idx, brk_l_cls)
            # clusters.insert(brk_idx+1, brk_r_cls)
            d_mat = np.delete(d_mat, brk_idx, axis=0)
            d_mat = np.delete(d_mat, brk_idx, axis=1)

            ## Add brk_L
            d_all_L, d_all_R = np.array([]), np.array([])
            for cl in clusters:
                d_add = __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_l_cls], linkage_method) if (__min_dist_cl(cl, brk_l_cls)) == 1 else MAX_DIST
                # d_all_L = np.append(d_all_L, __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_l_cls], linkage_method))
                d_all_L = np.append(d_all_L, d_add)
            d_mat = np.insert(d_mat, brk_idx, d_all_L, axis=0)
            d_all_L = np.insert(d_all_L, brk_idx, MAX_DIST)
            d_mat = np.insert(d_mat, brk_idx, d_all_L, axis=1)
            clusters.insert(brk_idx, brk_l_cls)

            ## Add brk_R
            for cl in clusters:
                d_add = __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_r_cls], linkage_method) if (__min_dist_cl(cl, brk_r_cls)) == 1 else MAX_DIST
                # d_all_R = np.append(d_all_R, __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_r_cls], linkage_method))
                d_all_R = np.append(d_all_R, d_add)
            d_mat = np.insert(d_mat, brk_idx+1, d_all_R, axis=0)
            d_all_R = np.insert(d_all_R, brk_idx+1, MAX_DIST)
            d_mat = np.insert(d_mat, brk_idx+1, d_all_R, axis=1)                
            clusters.insert(brk_idx+1, brk_r_cls)
            
        else:
            d_mat = np.delete(d_mat, brk_idx, axis=0)
            d_mat = np.delete(d_mat, brk_idx, axis=1)

            ## Add brk_L
            d_all_L, d_all_R = np.array([]), np.array([])
            for cl in clusters:
                d_all_L = np.append(d_all_L, __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_l_cls], linkage_method))
            d_mat = np.insert(d_mat, brk_idx, d_all_L, axis=0)
            d_all_L = np.insert(d_all_L, brk_idx, MAX_DIST)
            d_mat = np.insert(d_mat, brk_idx, d_all_L, axis=1)
            clusters.insert(brk_idx, brk_l_cls)

            ## Add brk_R
            for cl in clusters:
                d_all_R = np.append(d_all_R, __compute_dist_cls([seqs[id] for id in cl], [seqs[id] for id in brk_r_cls], linkage_method))
            d_mat = np.insert(d_mat, brk_idx+1, d_all_R, axis=0)
            d_all_R = np.insert(d_all_R, brk_idx+1, MAX_DIST)
            d_mat = np.insert(d_mat, brk_idx+1, d_all_R, axis=1)                
            clusters.insert(brk_idx+1, brk_r_cls)

        # print(f'Z_ID CHK: {prev_Z_idx}, Z: {len(Z)}, num: {num_seq}, del: {Z[prev_Z_idx]}')
        del Z[prev_Z_idx]   ## del Z old brk and add_pair
        for z_id in range(prev_Z_idx,len(Z)):
            if Z[z_id][0] > prev_Z_idx+num_seq: Z[z_id][0] = Z[z_id][0]-1
            if Z[z_id][1] > prev_Z_idx+num_seq: Z[z_id][1] = Z[z_id][1]-1
            
        del add_pair[prev_Z_idx]    ## del break-cluster

        if brk_Z_idx >= len(Z)+num_seq: brk_Z_idx -=1
        
        # print(f'[L-R]: {brk_l_cls}-{brk_r_cls} and [D remain-L]: {d_res_L}, [D remain-R]: {d_res_R}, d_brk: {d_brk}')

        # print(f'[REVISE] RES:{res_cls}, BRK: {brk_cls}')
        if res_idx == right: 
            left, right, c1, d_c1 = clusters.index(brk_cls), clusters.index(res_cls), brk_Z_idx, d_brk
            ## c2, d_c2= res, d_c2 (remain)
        else: 
            right, left, c2, d_c2 = clusters.index(brk_cls), clusters.index(res_cls), brk_Z_idx, d_brk
            ## c1, d_c1= res, d_c1 (remain)

        if left > right: left, right, c1, c2, d_c1, d_c2 = right, left, c2, c1, d_c2, d_c1  ## refine for cascade-flip
        
        if c1 >= prev_Z_idx+num_seq: c1 -=1
        if c2 >= prev_Z_idx+num_seq: c2 -=1

        # if kadj==1:
            # c3, c4 = d_mat[left], len(res_cls) + len(brk_cls)
        # else:
            # c3, c4 = d_mat[left, right], len(res_cls) + len(brk_cls)
        c3, c4 = __compute_dist_cls([seqs[id] for id in res_cls], [seqs[id] for id in brk_cls], linkage_method), len(res_cls) + len(brk_cls)

    # print(f'[Revised] d_c1: {d_c1}, d_c2: {d_c2}, c3: {c3}')
        
    add_idx = len(d_mat)+1 if kadj ==1 else len(d_mat)

    # print(f'After Flipped: lenCL: {len(clusters)}, lenD: {len(d_mat)}, LR: {left}-{right}')
    return left, right, c1, c2, c3, add_idx, d_mat, clusters, Z, add_pair

#  @params sel_idx: selected idx of the subsequence (to find the hierarchical tree)
#  @params Z: matrix Z of hierarchical clustering 
#  @params add_pair: history of added clusters for each step
#  @params num_seq: num_seq at the start of clustering (to chk Z idx)
def __get_upper_cl(sel_idx, num_seq, Z, add_pair):
    """
    To cut the tree, finding upper (sub) tree of selected subseq.
    """
    Z_part, cl_part = [], []
    idx_z = []
    for i, z_i in enumerate(Z):
        if z_i[0] == sel_idx or z_i[1] == sel_idx:
            Z_part.append(z_i)
            cl_part.append(add_pair[i])
            idx_z.append(i)
            sel_idx = i+num_seq
            # print('[CLS]', sel_idx, z_i)
    Z_part = np.array(Z_part)
    return Z_part, cl_part, idx_z

#  @params sel_idx: selected idx of the subsequence (to find the hierarchical tree)
#  @params Z: matrix Z of hierarchical clustering 
#  @params add_pair: history of added clusters for each step
#  @params num_seq: num_seq at the start of clustering (to chk Z idx)
def __get_lower_cl(sel_idx, num_seq, Z):
    Z_l, Z_r = Z[sel_idx][0], Z[sel_idx][1]
    while Z_l > num_seq and Z_r > num_seq:
        sel_idx = int(Z_l)-num_seq if Z[int(Z_l)-num_seq][2] > Z[int(Z_r)-num_seq][2] else int(Z_r)-num_seq
        Z_l, Z_r = Z[sel_idx][0], Z[sel_idx][1]
    Z_idx = Z_l if Z_l < num_seq else Z_r
    # print(f'Found: {Z_idx}')
    return int(Z_idx)

def __get_cont_cl(arr, gap, min_len):
    arr = np.asarray(arr)
    # mask = (arr ==1).astype(int)    ## set 0 if the value is not 1
    mask = (arr > 0) & (arr <= gap+1)
    if len(mask) ==0:
        return []

    diff = np.diff(mask.astype(int))
    start_i = np.where(diff ==1)[0] +1  ## start point of 1
    end_i = np.where(diff == -1)[0]     ## end point of 1

    ## exceptions
    if mask[0]:
        start_i = np.insert(start_i, 0, 0)
    if mask[-1]:
        end_i = np.append(end_i, len(arr)-1)

    lengths = end_i - start_i +1

    if len(lengths) == 0:
        return []

    valid_is = np.where(lengths > min_len)[0]
    return [(start_i[i], end_i[i]) for i in valid_is]
    # max_idx = np.argmax(lengths)
    # return start_i[max_idx], end_i[max_idx]

def __rev_side_cl(cl, outlier, d_seq_m, all_cl, seqs, l2):
    out_t = copy.deepcopy(outlier)
    cl_tmp = copy.deepcopy(cl)
    while (out_t[0] == cl_tmp[0]):
        if out_t[0] == 0: break

        if out_t[0] == cl_tmp[0]:
            l_m_subseq = None
            for i, cl_t in enumerate(all_cl):
                if out_t[0]-1 in cl_t:
                    l_m_subseq = np.mean([seqs[c] for c in cl_t], axis=0)
                    break
            if l_m_subseq is None: break
            # print('[L]', cl_t)
            if np.linalg.norm(compute_diff_dist(l_m_subseq, seqs[out_t[0]])) < d_seq_m[0]:
                cl_t.append(out_t[0])
                # print(f'Left: {out_t[0]} {l2[out_t[0]]} --> {out_t[0]-1} {l2[out_t[0]-1]}')
                l2[out_t[0]] = l2[out_t[0]-1]
                del cl_tmp[0]
                del out_t[0]
                if len(out_t) ==0: break
            else:
                break
        else: break
    while(out_t[-1] == cl_tmp[-1]):
        if out_t[-1] >= len(seqs)-1: break

        if out_t[-1] == cl_tmp[-1]:
            r_m_subseq = None
            for i, cl_t in enumerate(all_cl):
                if out_t[-1]+1 in cl_t:
                    r_m_subseq = np.mean([seqs[c] for c in cl_t], axis=0)
                    break
            if r_m_subseq is None: break
            if np.linalg.norm(compute_diff_dist(r_m_subseq, seqs[out_t[-1]])) < d_seq_m[-1]:
                cl_t.append(out_t[-1])
                # print(f'Right: {out_t[-1]} {l2[out_t[-1]]} --> {out_t[-1]+1} {l2[out_t[-1]+1]}')
                l2[out_t[-1]] = l2[out_t[-1]+1]
                del cl_tmp[-1]
                del out_t[-1]
                if len(out_t) ==0: break
            else:
                break
        else: break
    
    if len(cl_tmp) < len(cl):
        for cl_t in all_cl:
            if cl[int(len(cl)/2)] in cl_t:
                cl_t = cl_tmp
                # print(f'[B]: {cl} --> {cl_tmp}')
                return True

    else:
        return None
    

def __refine_clusters(cl_new, seqs, gap, max_W, l2):
    ## Refine clusters
    d_mats, d_intras, cl_outliers = [], [], []
    cl_refine = []
    l_new = copy.deepcopy(l2)
    cl_num = len(cl_new)
    # print(f'Check: {cl_num}, {np.max(l2)}')
    # print('TH:', np.min(size_cl))
    for cl in cl_new:
        cl_subseqs = []
        for c in cl: cl_subseqs.append(seqs[c])
        m_subseq = np.mean(cl_subseqs, axis=0)

        d_seq_m = [np.linalg.norm(compute_diff_dist(seq, m_subseq)) for seq in cl_subseqs]
        cl_out = [cl[i] for i, d_m in enumerate(d_seq_m) if d_m > np.mean(d_seq_m)+np.std(d_seq_m)]

        # print(f'Out: {cl_out}')
        ## Find 'continuous' outliers for checking division of subcluster
        cont_cl = np.array(cl_out[1:]) - np.array(cl_out[:-1])
        range_se = __get_cont_cl(cont_cl, gap, max_W)
        # print(f'gap: {math.ceil(np.min(size_cl)*0.1)}')
        if len(range_se) == 0:
            cl_refine.append(cl)
            # d_mats.append(distance.squareform(distance.pdist(cl_subseqs)))
            # sns.heatmap(d_mats[-1])
            # plt.show()
            continue
        
        cl_remain = copy.deepcopy(cl)
        for (s, e) in range_se:
            s, e = cl_out[s], cl_out[e+1]
            # if e-s+1 >2: print('continuous: ', e-s+1)

            # print(f'Min Size: {np.min(size_cl)*0.7} or {len(cl)/4}')
            ## Possible sub-clusters should have length longer than 70% of the minimum cluster or a quarter of itself
            # if (e-s+1) >= min(np.min(size_cl)*0.7, len(cl_remain)/4) and (e-s+1) >2:
            if (e-s+1) >= max_W and (e-s+1) >gap:
                # print(f'From {s} To {e} via {e-s+1}/{len(cl)}')
                tmp_cl = [seqs[c] for c in np.arange(s-1, e+1)]    ## add +- 1 subseqs
                d_adj = [np.linalg.norm(compute_diff_dist(tmp_cl[i], tmp_cl[i+1])) for i in range(len(tmp_cl)-1)]
                d_adj_cl = [np.linalg.norm(compute_diff_dist(seqs[cl[i]], seqs[cl[i+1]])) for i in range(len(cl)-1)]
                # print(f'New ADJ dist: {d_adj} vs. {np.mean(d_adj_cl) + np.std(d_adj_cl)}')
                id_refine = [i for i, d_a in enumerate(d_adj) if d_a < np.mean(d_adj_cl) + np.std(d_adj_cl)]
                # print(f'Refine idx: {id_refine}')
                if len(id_refine) >= len(d_adj)*0.7:
                    s = s-1+id_refine[0]
                    e = s+id_refine[-1]+1
                    if e >= len(seqs): e = len(seqs)-1
                    cl_refine.append(np.arange(s, e))
                    # if s != cl[0] and e != cl[-1]:
                        # __rev_side_cl(cl, cl_out, d_seq_m, cl_new, seqs, l2)

                    cl_num+=1
                    for c in cl_refine[-1]: l_new[c] = cl_num
                    cl_remain = [c for c in cl_remain if c not in np.arange(s, e)]
                    # cl_refine.append([c for c in cl if c not in np.arange(s, e)])
                    # print(f'DIV: {np.arange(s, e)}')
                    # print(f'remain: {[c for c in cl if c not in np.arange(s, e)]}')
            # else:
                # __rev_side_cl(cl, cl_out, d_seq_m, cl_new, seqs, l2)
                # cl_refine.append(cl)
        if len(cl_remain) > 0:
            __rev_side_cl(cl_remain, cl_out, d_seq_m, cl_new, seqs, l2)
            cl_refine.append(cl_remain)

        # d_mats.append(distance.squareform(distance.pdist(cl_subseqs)))
        # d_intras.append([np.linalg.norm(compute_diff_dist(seq, m_subseq)) for seq in cl_subseqs])
# 
        # sns.heatmap(d_mats[-1])
        # plt.show()
        # plt.plot(d_intras[-1])
        # plt.hlines(np.mean(d_intras[-1]),0, len(d_intras[-1]), color='r')
        # plt.hlines(np.mean(d_intras[-1])+np.std(d_intras[-1]),0, len(d_intras[-1]), color='g')
        # plt.show()

    return cl_refine, l_new

#  @params Z: matrix Z of hierarchical clustering 
#  @params add_pair: history of added clusters for each step
#  @params num_seq: num_seq at the start of clustering (to chk Z idx)
#  @params t_seqs: all subsequences 
def __cut_subtree(num_seq, Z, add_pair, t_seqs, isTrain, gap, max_W, cut, REFINE):
    """
    Multiple-cut of Z (dendrogram)
    """
    l2 = np.ones(num_seq)*(-1)
    d_z = Z[:,2]
    cl_new = []
    # print('For all Z:', np.mean(abs(d_z[1:]-d_z[:-1])))
    # print(f'Len Z: {len(Z)}, Len Seq: {len(t_seqs)}')

    all_cuts, chk_dist, part_z = [], [], []
    all_dist_Z, size_cl = [], []
    for idx in np.argsort(d_z):
        sel_idx = int(Z[idx][0]) if Z[idx][0] < len(t_seqs) else int(Z[idx][1])
        if sel_idx >= len(t_seqs): continue
        if l2[sel_idx] != -1: continue

        Z_part, cl_part, idx_z = __get_upper_cl(sel_idx, num_seq, Z, add_pair)
        # print(f'[IDX Z]: {idx_z}')
        diff_part_z = abs(Z_part[1:, 2] - Z_part[:-1,2])
        # diff_part_z = abs(Z_part[1:-1, 2] - Z_part[:-2,2])
        # print(f'ID: {sel_idx}, diff: {diff_part_z}, AVG: {np.mean(diff_part_z)}')
        if cut == None:
            th_diff = np.mean(diff_part_z)
        else:
            d_sub = diff_part_z[:[i for i, d_i in enumerate(diff_part_z) if d_i > np.mean(diff_part_z)][0]]
            d_step = (np.mean(diff_part_z) - np.mean(d_sub))/5
            th_diff = np.mean(diff_part_z) - d_step*cut
        
        # print('(1) diff_Z', diff_part_z)
        # print('(2) Z_d', Z_part[:,2])
        # all_cuts.append(th_diff)
        chk_dist.append(diff_part_z)
        part_z.append(Z_part[:,2])
        for i in range(1, len(diff_part_z)):                
            # print(f'th: {th_diff}, cut: {cut}, m: {np.mean(diff_part_z)}, m_sub: {np.mean(d_sub)}, step: {d_step}')
            # if  diff_part_z[i] > np.mean(diff_part_z):
            if diff_part_z[i] > th_diff:
#                if isTrain:
#                    ## Need to check the opposite subtree
#                    # print(f'** {Z_part[i]}')
#                    if int(Z_part[i][0]) in np.array(idx_z)+num_seq:
#                        out_idx = Z_part[i][1]
#                    else:
#                        out_idx = Z_part[i][0]
#                    # print(f'(2) out idx: {out_idx}')
#                    ## pick one of the opposite tree member, and apply cut
#                    Z_idx_t = __get_lower_cl(int(out_idx)-num_seq, num_seq, Z)
#                    print('[Z_IDX]', Z_idx_t, out_idx)
#                    Z_part_l, cl_part_l, idx_z_l = __get_upper_cl(Z_idx_t, num_seq, Z, add_pair)
#                    diff_part_z_l = abs(Z_part_l[1:,2] - Z_part_l[:-1,2])
#                    # diff_part_z_l = abs(Z_part_l[1:-1,2] - Z_part_l[:-2,2])
#                    # print(f'Out: {out_idx}, Picked: {Z_idx_t}, Ths: {np.mean(diff_part_z_l)} vs. {th_diff}')
#                    for j in range(1, len(diff_part_z_l)):
#                        if diff_part_z_l[j] > np.mean(diff_part_z_l): break
#                    # print(f'Compare: {Z_part_l[:,2][j]} vs. {Z_part[:,2][i]}')
#
#                    all_cuts.append(th_diff)
#                    all_dist_Z.append(Z_part[:,2][i])
#                    # print('(3) diff_z_L:', diff_part_z_l)
#                    # print('(4) Z-d_L:', Z_part_l[:,2])
#
#                    ## Divide cluster here first
#                    if Z_part_l[:,2][j] < Z_part[:,2][i]:
#                        cl_tmp_l = copy.deepcopy(cl_part_l[j])
#                        print('Div_Init: ', cl_tmp_l)
#                        for c in cl_part_l[j]:
#                            if l2[int(c)] != -1:
#                                cl_tmp_l.remove(c)
#                            else:
#                                l2[int(c)] = len(cl_new) +1
#                        if len(cl_tmp_l) >0:
#                            cl_new.append(cl_tmp_l)
#                            print("Divide L:", cl_tmp_l)
                
                ## Cut sub-tree here
                cl_tmp = copy.deepcopy(cl_part[i])
                for c in cl_part[i]:
                    if l2[int(c)] != -1: 
                        cl_tmp.remove(c)
                    else:
                        l2[int(c)] = len(cl_new)+1

                cl_new.append(cl_tmp)
                # print('CL:', cl_tmp)

                size_cl.append(len(cl_tmp))
                all_cuts.append(th_diff)
                all_dist_Z.append(Z_part[:,2][i])
                
                # for id in cl_tmp: l2[id]=len(cl_new)
                # print('MCUT:', cl_tmp, diff_part_z[i], np.mean(diff_part_z), np.std(diff_part_z[:i]), np.mean(diff_part_z)+np.std(diff_part_z[:i]))
                break
    ## Refine clusters
    if isTrain and REFINE:
        # print('Before ', set(l2))
        cl_refine, l_new = __refine_clusters(cl_new, t_seqs, gap, max_W/2, l2)
        # print('After ', set(l_new))
        # print(f'CL NUMs: {len(cl_new)} --> {len(cl_refine)}')
        # for cl in cl_refine:
            # print(f'CL N: {cl}')
        
        return l_new, cl_refine
    else:
        return l2, cl_new

#  @params seq: subsequence to test
#  @params m_subseq: normal pattern to compare
#  @params m_tau: tau threshold for the m_subseq
#  @params eta: exponential parameter eta
def membership(seq, m_subseq, m_tau, eta):
    """
    Compute the membership of given seq for the m_subseq pattern
    """
    d = np.linalg.norm(compute_diff_dist(seq, m_subseq))
    if  d<= m_tau:
        return 1, d
    else:
        return math.exp(-eta*(d-m_tau)), d

#  @params cl_id: cluster ID
#  @params Sw: window size for testing nu
#  @params listcluster: list of clusters (for the subseq)
def __range_Sw(cl_id, Sw, listcluster, isTrain):
    """
    To compute the nu, find the range of time-series that contain cl_ID
    """
    start, end = [], []
    inds = [j for j, l in enumerate(listcluster) if l==cl_id]
    inds.sort()
    # print('C IDs:', inds)
    start.append(int(np.max([0, inds[0]-Sw+1])))
    for i, ind in enumerate(inds[1:]):
        ## To find end
        if len(start) > len(end):
            if ind < inds[i]+Sw: 
                continue
            else: 
                # print('END?:' ,ind, inds[i], Sw, inds[i] +Sw)
                ## revise start
                if isTrain and start[-1] == 0: start[-1] = np.max([(ind-Sw+1)//2,0])
                end.append(int(ind+Sw-1))
        else:
            start.append(int(ind-Sw+1))
    ## Check the last ind
    if len(start) > len(end):
        if isTrain and start[-1] == 0: start[-1] = np.max([(inds[-1]-Sw+1)//2, 0])
        end.append(int(np.min([inds[-1]+Sw-1, len(listcluster)])))
        # if isTrain and end[-1] == len(listcluster) and end[-1] > start[-1]: end[-1] = (inds[-1]-Sw+1)
    
    # print('[S:E]', start, end)
    return start, end
    
#  @params seq_n: divided subsequences
#  @params Sw: window size for testing nu
#  @params m_subseq: normal pattern to compare
#  @params m_tau: tau threshold for the m_subseq
#  @params eta: exponential parameter eta
def __find_freq_th(seq_n, Sw, m_subseq, m_tau, eta):
    """
    Compute the minimum membership (nu) 
    """
    mem_t = []
    for seq in seq_n:
        mem_t.append(membership(seq, m_subseq, m_tau, eta)[0])
    avg_mem = []
    for i in range(len(seq_n)-Sw+1):
        avg_mem.append(np.mean(mem_t[i:i+Sw]))
    return np.mean(avg_mem)

def __get_freq_th(seqs, Sw, cl_ids, listcluster_rev, m_subseq, m_tau, eta, max_nu, isTrain):
    m_nu = []
    for j, cl_id in enumerate(cl_ids):
        inds = [k for k, l in enumerate(listcluster_rev) if l==cl_id]
        starts, ends = __range_Sw(cl_id, Sw, listcluster_rev, isTrain)
        nu_t = []
        # if isTrain == False:
            # print('NM:', j, '[Start:End]', starts, ends, inds)
        for s, e in zip(starts, ends):
            # if e-s >= 2*Sw:
            # if e-s >= Sw:
            range_seq = seqs[s:e]
            # print('[LEN LISTCLUSTER]', j, cl_id, set(list(listcluster)), len(m_subseq), len(m_tau))
            nu_t.append(__find_freq_th(range_seq, Sw, m_subseq[j], m_tau[j], eta))
            # print('? NU: ', nu_t)
        # if isTrain == False:
            # print("NU-all", nu_t)
        # if np.min(nu_t) > max_nu:
        if np.min(nu_t) > max_nu:
            m_nu.append(max_nu)
        elif np.min(nu_t) < 0.1:
            m_nu.append(0.1)
        else:
            m_nu.append(np.min(nu_t))
            # m_nu.append(np.mean(nu_t))

    return m_nu

#  @params seq_n: divided subsequences to cluster
#  @params linkage_method: linkage method for hierarchical clustering (ex. ward2)
#  @params th_reverse: % of difference to check flip
#  @params cut: user-parameter to cut the dendrogram??
#  @params kadj: k-adjacent distance to compare
#  @params eta: exponential parameter eta
#  @params max_W: maximum W size to allow
def adaptive_ahc(seqs, linkage_method='ward', th_reverse=5, cut=None, kadj=1, eta=1, max_W = 20, max_nu=0.9, min_size=0.025, isTrain=True, NMs =None, REFINE=True):
    """
    Adjacent linkage computing function
    """
    ## (1) Compute the linkage first
    clusters = [[i] for i in range(len(seqs))]  ## init.    
    Z, add_pair = [], []
    num_seq = len(seqs)

    # if isTrain: print(f'[CHK] #seq: {num_seq}, LEN: {len(seqs[0])}')

    # Compute all pairwise distances: Init.
    if kadj ==1:
        d_vec = __adj_dist(seqs, linkage_method)
        # print('VECTOR', d_vec)
        d_mat = np.ones((num_seq, num_seq))*MAX_DIST
        for i, d in enumerate(d_vec):
            d_mat[i, i+1] = d_mat[i+1, i] = d
        # print('MAT: ', d_mat)
    else:
        d_mat = distance.squareform(distance.pdist(seqs))
        for i in range(len(d_mat)): d_mat[i,i] = MAX_DIST ## for init. (inf. diagonal)

    add_idx = len(d_mat)   ## for Z, index init

    ## Merge until all
    while len(clusters) >1:
        ## Find two clusters are placed within a certain (k-adjacent) range
        left, right, c3 = __find_min(d_mat, kadj, clusters)
        # print(f'[RE-1]: left: {left}, right: {right}, lenCL: {len(clusters)}, lenD: {len(d_mat)}')
        c1, c2, d_c1, d_c2 = __get_index_Z(Z, add_pair, clusters, left, right, num_seq)

        # print(f'[Normal] [L-R] {left}, {right}, [C12] => {c1}, {c2}')
        # print(f'[Clusters] L: {clusters[left]} \n R:{clusters[right]}')
        ## Cascade flip-merge TODO: Reduce args
        if d_c1-c3 > d_c1*th_reverse/100 or d_c2-c3> d_c2*th_reverse/100:
            
            if __min_dist_cl(clusters[left], clusters[right]) < max(kadj, max_W):
                left, right, c1, c2, c3, add_idx, d_mat, clusters, Z, add_pair = flipped_merge(seqs, left, right, c1, c2, c3, d_c1, d_c2, Z, d_mat, clusters, add_pair, num_seq, th_reverse, kadj, max_W, linkage_method)
                # print(f'[AFTER FLIP] {c1}, {c2}, IDX {add_idx}')

        c4 = len(clusters[left]) + len(clusters[right])
        Z.append([int(c1), int(c2), c3, c4])

        # print('[D]', d_mat)
        # print(f"[Merge] left: ({left}) {clusters[left]}, right: ({right}) {clusters[right]}, d:{c3}")
        np.set_printoptions(precision=2)

        ## Update dist matrix, lrs, and clusters
        clusters, d_mat = __merge_cls(left, right, clusters, d_mat, seqs, kadj, linkage_method=linkage_method)
        new_cl = copy.deepcopy(clusters[left])
        # print('NEW CL:', new_cl)
        add_pair.append(new_cl)
        # print('[LEN CL]', len(clusters))
        add_idx +=1
        # print(f'[IDX] {add_idx}, {len(add_pair), len(clusters)} ==========================================================================')

    Z = np.array(Z)
    if len(Z) <=1:
        return None, None, None, None, None, None, None, None, None, None, None

    gap = kadj if kadj !=1 else 2
    listcluster, new_cls = __cut_subtree(len(seqs), Z, add_pair, seqs, isTrain, gap, max_W, cut=cut, REFINE=REFINE)

    cl_lengths = []
    for i in range(1, int(np.max(listcluster))+1):
        cl_lengths.append(len([j for j, l in enumerate(listcluster) if l==i]))

    ## For NM
    m_subseq, single_subseq, d_subseq = [], [], []
    cl_ids, nums = [], []
    Ws, m_tau, m_nu = [], [], []

    listcluster_rev = copy.deepcopy(listcluster)
    for i in range(1, int(np.max(listcluster))+1):
        inds = [j for j, l in enumerate(listcluster) if l==i]
        cl_is = [seqs[j] for j in inds]
        
        min_cl_size = np.sum(cl_lengths)*min_size if isTrain else np.sum(cl_lengths)*np.mean([nm_t.nu for nm_t in NMs]) #np.sum(cl_lengths)*(1-max_nu)
        # if isTrain == False: print(f'[is NaN?] {min_cl_size}, {[nm_t.nu for nm_t in NMs]}, {len(NMs)}')
        # print('CL LEN:', len(cl_is), np.sum(cl_lengths), set(list(listcluster)), isTrain, min_cl_size)
        # print('[LENGTH]: ', cl_lengths)
        ## Check th_size_nm here
        ## TEST JJ
        # elif len(cl_is) > min_cl_size and len(cl_is)*(max_nu) >1:
        if len(cl_is) >= int(min_cl_size) and len(cl_is)*(max_nu) >1:
            
            ## take average for the group (for 90%, most similar ones)
            # m_subseq.append(np.mean(cl_is, axis=0))
            # d_subseq.append([compute_diff_dist(m_subseq[-1], np.array(t_subseq)) for t_subseq in cl_is])
            tmp_m = np.mean(cl_is, axis=0)
            tmp_dist = [np.linalg.norm(compute_diff_dist(tmp_m, np.array(t_subseq))) for t_subseq in cl_is]
            
            sel_seqs = [cl_is[tmp_i] for ii, tmp_i in enumerate(np.argsort(tmp_dist)) if ii <= len(cl_is)*(max_nu)]
            m_subseq.append(np.mean(sel_seqs, axis=0))
            d_subseq.append([compute_diff_dist(m_subseq[-1], np.array(t_subseq)) for t_subseq in sel_seqs])
            
            cl_ids.append(i)
            nums.append(len(cl_is))
            Ws.append(longest_consecutive_sequence(inds))
        else:
            for k in inds:
                listcluster_rev[k] = -1
            if len(cl_is) == 1:
                # listcluster_rev[inds[0]] = -1
                single_subseq.append(cl_is[0])

    # print('After Checking:', set(list(listcluster_rev)), len(m_subseq))
    if len(m_subseq) == 0:
        return None, None, None, None, None, None, None, None, None, None, None
    ## Computing Stats.
    d_ci, d_c_std = intra_cluster_dist(d_subseq)
    # print('NUM_CL:', len(m_subseq), len(d_subseq), len(d_ci))
    m_tau = (d_ci+3*d_c_std)

    if isTrain:
        Sw = np.min([max_W, 2*np.max(Ws)])
    else:
        Sw = max_W
    if Sw > len(seqs): Sw = len(seqs)
    print('SW:', Sw)

    m_nu = __get_freq_th(seqs, Sw, cl_ids, listcluster_rev, m_subseq, m_tau, eta, max_nu, isTrain)
#    for j, cl_id in enumerate(cl_ids):
#        inds = [k for k, l in enumerate(listcluster_rev) if l==cl_id]
#        starts, ends = __range_Sw(cl_id, Sw, listcluster_rev)
#        nu_t = []
#        print('NM:', j, '[Start:End]', starts, ends, inds)
#        for s, e in zip(starts, ends):
#            # if e-s >= 2*Sw:
#            # if e-s >= Sw:
#            range_seq = seqs[s:e]
#            # print('[LEN LISTCLUSTER]', j, cl_id, set(list(listcluster)), len(m_subseq), len(m_tau))
#            nu_t.append(__find_freq_th(range_seq, Sw, m_subseq[j], m_tau[j], eta))
#            # print('? NU: ', nu_t)
#        # print("NU-all", nu_t)
#        # if np.min(nu_t) > max_nu:
#        if np.mean(nu_t) > max_nu:
#            m_nu.append(max_nu)
#        elif np.mean(nu_t) < 0.1:
#            m_nu.append(0.1)
#        else:
#            # m_nu.append(np.min(nu_t))
#            m_nu.append(np.mean(nu_t))

    set_cluster = list(set(list(listcluster_rev)))
    
    if -1 in set_cluster:
        set_cluster.remove(-1)
    for i, set_cl in enumerate(set_cluster):
        ids = [j for j, c in enumerate(listcluster_rev) if c==set_cl]
        for id in ids: listcluster_rev[id] = i
    
    
    return listcluster_rev, Z, m_subseq, m_tau, m_nu, single_subseq, d_subseq, d_ci, d_c_std, Ws, add_pair
