import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.TSB_AD.metrics import metricor
import sys

# Return the intervals where there is an anomaly
# @param y: ndarray of shape (N,) corresponding to anomaly labels
# @Return list of lists denoting anomaly intervals in the form [start, end)
def find_anomaly_intervals(y):
    """
    Update the Ward-distance vector of adjacent clusters
    """
    change_indices = np.where(np.diff(y) != 0)[0]
    if len(change_indices) == 0:
        return []
    anom_intervals = []

    if y[change_indices[0]] == 0:
        i = 0
    else:
        i = 1
        anom_intervals.append([0,change_indices[0]+1])

    while (i + 1 < len(change_indices)):
        anom_intervals.append([change_indices[i]+1,change_indices[i+1]+1])
        i += 2

    if y[-1] == 1:
        anom_intervals.append([change_indices[-1]+1,len(y)])

    return anom_intervals

## Return: None (draw plot)
## @param X: time-series, ndarray (N,)
## @param y: label, ndarray (N,)
## @param start: start-point of plot
## @param end: end-point of plot
## @param title: title of plot
## @param: marker: marker for time-series
def plot_anomaly(X, y, start=0, end=sys.maxsize, title="", marker="-"):
    # Plot the data with highlighted anomaly
    plt.figure(figsize=(14,2))
    plt.plot(np.arange(start,min(X.shape[0],end)), X[start:end], f"{marker}b")
    for (anom_start, anom_end) in find_anomaly_intervals(y):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'r-')
            # print(anom_start, anom_end)
    if len(title) > 0:
        plt.title(title)

## Return: the intervals for each clusters
## @param y: ndarray of of shape (cl,) corresponding to list_clusters
def find_cluster_intervals(y):
    """
    To draw clusters in different colors
    """
    num_cl = int(np.max(y))+1
    change_indices = np.where(np.diff(y) != 0)[0]
    if len(change_indices) == 0:
        return []
    cl_intervals = []

    for cl in range(num_cl):
        cl_interval = []
        for i in range(len(change_indices)):
            if y[change_indices[i]] == cl:
                if i==0:
                    cl_interval.append([0, change_indices[i]+1])
                else:
                    cl_interval.append([change_indices[i-1]+1, change_indices[i]+1])
        
        if y[-1] == cl:
            cl_interval.append([change_indices[-1]+1,len(y)])

        cl_intervals.append(cl_interval)
    return cl_intervals

## Return: None (draw plot)
## @param X: time-series, ndarray (N,)
## @param y: label, ndarray (N,)
## @param cls: cluster for each point, ndarray (N,) 
## @param ths: thresholds for each point, ndarray (N,)
## @param start: start-point of plot
## @param end: end-point of plot
## @param title: title of plot
## @param: marker: marker for time-series
## @param size_x and size_y: size of fig
def plot_training(X, y, scores, cls, ths, start=0, end=sys.maxsize, title="", marker="-", size_x =12, size_y=5):
    """
    Plot for training data (results of anomaly detection)
    """
    # Plot the data with highlighted anomaly
    fig1 = plt.figure(figsize=(size_x, size_y), constrained_layout=True)
    gs = fig1.add_gridspec(2, 4)
    
    f1_ax1 = fig1.add_subplot(gs[0,:])
    plt.tick_params(labelbottom=False)

    plt.plot(np.arange(start,min(X.shape[0],end)), X[start:end], f"{marker}b")
    for (anom_start, anom_end) in find_anomaly_intervals(y):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'rx-')
            # print(anom_start, anom_end)
    if len(title) > 0:
        plt.title(title)

    f1_ax2 = fig1.add_subplot(gs[1,:])
    plt.plot(scores, label='Score')
    plt.plot(ths, 'r:', label='TH')
    # plt.plot(y*ths[0]/2, label='Label')
    # plt.plot(cls*ths[0]/3, label='Cluster')
    plt.legend()

## Return: None (draw plot)
## @param X: time-series, ndarray (N,)
## @param label: label, ndarray (N,)
## @param cls: cluster for each point, ndarray (N,) 
## @param ths: thresholds for each point, ndarray (N,)
## @param start: start-point of plot
## @param end: end-point of plot
## @param title: title of plot
## @param: marker: marker for time-series
## @param size_x and size_y: size of fig
## @param ylim: ylim for fig
## @param lx, ly, ncol: for legend position
def plot_cluster(X, cls, label, start=0, end=sys.maxsize, title="", marker="-", size_x=12, size_y=2, ylim=None, lx=0.5, ly=1.5, ncol=None):
    """
    Plot clusters of each dataset: color-coded
    """
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 
          'blue', 'orange', 'green', 'purple','brouwn', 'gold', 'violet', 'cyan']
    plt.figure(figsize=(size_x,size_y))
    plt.plot(np.arange(start, min(X.shape[0], end)), X[start:end], f"{marker}k")
    i=0
    for cl_i in find_cluster_intervals(cls):
        print(cl_i)
        for (cl_start, cl_end) in cl_i:
            if start <= cl_end and cl_start <= cl_end:
                cl_start = max(start, cl_start)
                cl_end = min(end, cl_end)
                plt.plot(np.arange(cl_start, cl_end), X[cl_start:cl_end], color=colors[i])
        plt.plot(np.arange(cl_start, cl_end), X[cl_start:cl_end], color=colors[i], label=f'Cluster: {i}')
        i +=1
    for (anom_start, anom_end) in find_anomaly_intervals(label):
        if start <= anom_end and anom_start <= anom_end:
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'r-')
            # print(anom_start, anom_end)
    plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'r-', label='Anomaly')
    if len(title) >0: plt.title(title)
    if ylim is not None: plt.ylim(ylim)
    if ncol is None:
        ncol = i+1
    plt.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(lx, ly))

## Return: None (draw plot)
## @param X: time-series, ndarray (N,)
## @param label: label, ndarray (N,)
## @param cls: cluster for each point, ndarray (N,) 
## @param ths: thresholds for each point, ndarray (N,)
## @param start: start-point of plot
## @param end: end-point of plot
## @param title: title of plot
## @param: marker: marker for time-series
## @param size_x and size_y: size of fig
## @param ylim: ylim for fig
## @param lx, ly, ncol: for legend position
def plot_cluster_color(X, cls, label, start=0, end=sys.maxsize, title="", marker="-", size_x=12, size_y=2, ylim=None, lx=0.5, ly=1.5, ncol=None):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:gray', 
              'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan', 'pink', 'deepskyblue', 'lawngreen',
              'royalblue', 'darkgrey', 'darkorange', 'darkgreen','darkviolet','salmon','olivedrab','lightcoral','darkcyan','yellowgreen']
    # colors = ['tab:orange', 'tab:olive', 'tab:purple', 'tab:pink', 'tab:blue', 'tab:pink', 'tab:green', 'tab:cyan', 
        #   'blue', 'orange', 'green', 'purple','brown', 'gold', 'violet', 'cyan']
    plt.figure(figsize=(size_x,size_y))
    plt.plot(np.arange(start, min(X.shape[0], end)), X[start:end], f"{marker}k")

    # len_cl_i = []
    # cl_is = find_cluster_intervals(cls)
    # for cl_i in cl_is: len_cl_i.append(len(cl_i))
    i=0
    for cl_i in find_cluster_intervals(cls):
    # print(len_cl_i)
    # for ind in np.argsort(len_cl_i)[::-1]:
        # cl_i = cl_is[ind]
        # print(cl_i)
        if len(cl_i) ==0: 
            i+=1
            continue
        else:
            for (cl_start, cl_end) in cl_i:
                if start <= cl_end and cl_start <= cl_end:
                    cl_start = max(start, cl_start)
                    cl_end = min(end, cl_end)
                    # print(i, colors[i])
                    plt.plot(np.arange(cl_start, cl_end), X[cl_start:cl_end], color=colors[i])
            plt.plot(np.arange(cl_start, cl_end), X[cl_start:cl_end], color=colors[i], label=f'Cluster: {i}')
            i +=1
    if len([x for x in range(len(label)) if label[x] ==1]) !=0:
        for (anom_start, anom_end) in find_anomaly_intervals(label):
            if start <= anom_end and anom_start <= anom_end:
                anom_start = max(start, anom_start)
                anom_end = min(end, anom_end)
                plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'r-')
                # print(anom_start, anom_end)
        plt.plot(np.arange(anom_start, anom_end), X[anom_start:anom_end], 'r-', label='Anomaly')
    if len(title) >0: plt.title(title)
    if ylim is not None: plt.ylim(ylim)
    if ncol is None:
        ncol = i+1
    # plt.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(lx, ly))


################################################################################################################################    
def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None, y_pred=None, th=None):
    grader = metricor()
    
    if np.sum(label) != 0:
        R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True, ths=th) #
    
        L, fpr, tpr= grader.metric_new(label, score, plot_ROC=True, ths= th)
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
    ind_result = ['auc', 'precision', 'recall', 'f', 'Rrecall', 'ExistenceReward', 'OverlapReward', 'Rprecision', 'Rf', 'precision_at_k']

    L[0] = R_AUC
    # print(L)
    df = pd.DataFrame(L, index=ind_result, columns=['A2D2'])
    
    return df

def plotFigRev(data, label, scores, slabels, slidingWindow, plotRange=None, y_pred=None, th=None, th_addd = None, fname='temp.png'):
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
    # if len(data) < se[2]:
        # plt.plot(range(se[2], se[3]), data[se[2]:se[3]],'b', label='Normal-2')
    # if isinstance(se[0], int):        
        # if se[0] > 0:
            # plt.plot(range(se[0], se[1]), data[se[0]:se[1]], 'k')
    # elif len(se[0]) >1:
        # print('CHK:', se[0])
        # for s1, s2 in zip(se[0], se[1]):
            # plt.plot(range(s1, s2), data[s1:s2], 'k')
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
    
    print('CHK NUM_MODEL:', len(scores))
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

    plt.show()
    # plt.savefig(fname, bbox_inches='tight')
    # plt.savefig('two_normal_local.png', bbox_inches='tight')

def plot_membership(clf, figsize=(14,2), delay=0):
    plt.figure(figsize=figsize)
    for i, mem in enumerate(clf.mem_nm):
        # print(len(mem))
        if len(mem) < len(clf.mem_nm[0]):
            mem2 = mem[delay:]
            mem2 = np.pad(mem, (len(clf.mem_nm[0])-len(mem), 0), 'constant', constant_values=0)
            # print(len(mem2))
        else: mem2 = mem
        plt.plot(mem2, '.-', label=f'mem {i}')
        # if i ==1: break
    plt.show()