import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from util.TSB_AD.metrics import metricor
import sys
import matplotlib.dates as mdates


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
    plt.legend(ncol=ncol, loc='upper center', bbox_to_anchor=(lx, ly))


################################################################################################################################    
def plotFigRev(data, label, scores, slabels, slidingWindow, plotRange=None, y_pred=None, th=None, th_addd = None, fname='temp.png'):
    grader = metricor()
    
    range_anomaly = grader.range_convers_new(label)
    max_length = len(label)

    if plotRange==None:
        plotRange = [0,max_length]
    
    fig3 = plt.figure(figsize=(20, 3*len(scores)+4), constrained_layout=True)
    gs = fig3.add_gridspec(len(scores)+1, 4)
    
    f3_ax1 = fig3.add_subplot(gs[0, :])
    plt.tick_params(labelbottom=False)
   
    plt.plot(data,'k', label='Normal-1')
    if np.any(y_pred):
        plt.plot(y_pred[:max_length], 'c')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r', label='Anomalies')
    plt.xlim(plotRange)
    plt.legend(bbox_to_anchor=(0.5, 1.4), ncols=3, loc='upper center', facecolor='None')
    
    print('CHK NUM_MODEL:', len(scores))
    for i, score in enumerate(scores):
        f3_ax2 = fig3.add_subplot(gs[1+i, :])
        if len(score) < len(label):
            padded = np.full(max_length, np.nan)
            padded[-len(score):] = score
            plt.plot(padded, label=slabels[i])
        else:
            plt.plot(score[:max_length], label=slabels[i])
        if th is None:
            plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
        else:
            if i == 3:                
                plt.plot(th_addd, 'r:')
            else:
                plt.hlines(th,0,max_length,linestyles='--',color='red')
        plt.ylabel('score')
        plt.xlim(plotRange)
        plt.legend(loc='lower right')

    # plt.show()
    return fig3

def plot_membership(
    clf,
    df_data=None,
    data=None,
    label=None,
    timestamp_col='timestamp',
    traffic_choose='total_flow',
    stride=24,
    start_date=None,
    end_date=None,
    figsize=None,
    delay=0,
    sel_mem = True,
):
    if sel_mem:
        mem_n = max(len(mem) for mem in clf.mem_nm)
    else:
        mem_n = max(len(mem) for mem in clf.dist_nm)

    mems = clf.mem_nm if sel_mem else clf.dist_nm
    drift_points = getattr(clf, 'drift_points', None)
    if df_data is None:
        if figsize is None:
            figsize = (14, 2)
        plt.figure(figsize=figsize)
        # mems = clf.mem_nm if sel_mem else clf.dist_nm
        # for i, mem in enumerate(clf.mem_nm):
        for i, mem in enumerate(mems):
            mem2 = np.asarray(mem, dtype=float)[delay:]
            if len(mem2) < mem_n:
                mem2 = np.pad(mem2, (mem_n - len(mem2), 0), 'constant', constant_values=0)
            else:
                mem2 = mem2[-mem_n:]
            plt.plot(mem2, '.-', label=f'mem {i}')
        if drift_points is not None and len(drift_points) > 0:
            drift_positions = np.asarray(drift_points, dtype=float) / stride
            drift_positions = drift_positions[(0 <= drift_positions) & (drift_positions < mem_n)]
            for drift_pos in drift_positions:
                plt.axvline(drift_pos, color='tab:purple', linestyle='--', lw=1.0, alpha=0.7)
        plt.legend()
        plt.show()
        return

    df_plot = df_data.copy()
    df_plot[timestamp_col] = pd.to_datetime(df_plot[timestamp_col])
    timestamps = df_plot[timestamp_col]

    if data is None:
        data = df_plot[traffic_choose].to_numpy()
    data = np.asarray(data)

    mask = np.ones(len(df_plot), dtype=bool)
    if start_date is not None:
        mask &= timestamps >= pd.to_datetime(start_date)
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        if isinstance(end_date, str) and len(end_date.strip()) <= 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= timestamps <= end_ts
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        raise ValueError(f'No data in selected range: {start_date} ~ {end_date}')

    mem_idx = np.arange(mem_n) * stride
    mem_idx = mem_idx[mem_idx < len(df_plot)]
    mem_time = timestamps.iloc[mem_idx]
    mem_mask = mask[mem_idx]

    n_rows = 1 + len(mems)
    if figsize is None:
        figsize = (14, 2.0 + 1.4 * len(mems))

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1.4] + [1] * len(mems)},
    )

    axes[0].plot(timestamps[mask], data[mask], 'k-', lw=1.2, label=traffic_choose)
    if label is not None:
        label = np.asarray(label)
        anom_mask = mask & (label == 1)
        axes[0].scatter(timestamps[anom_mask], data[anom_mask], color='red', s=18, zorder=3, label='label = 1')
    axes[0].set_ylabel(traffic_choose)
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.25)

    for i, mem in enumerate(mems, start=1):
        mem2 = np.asarray(mem, dtype=float)[delay:]
        if len(mem2) < mem_n:
            mem2 = np.pad(mem2, (mem_n - len(mem2), 0), 'constant', constant_values=0)
        else:
            mem2 = mem2[-mem_n:]
        mem2 = mem2[:len(mem_idx)]
        axes[i].plot(mem_time[mem_mask], mem2[mem_mask], '.-', lw=1.0, ms=3)
        axes[i].set_ylabel(f'mem {i - 1}')
        axes[i].grid(alpha=0.25)

    if drift_points is not None and len(drift_points) > 0:
        drift_idx = np.asarray(drift_points, dtype=int)
        drift_idx = drift_idx[(0 <= drift_idx) & (drift_idx < len(df_plot))]
        drift_idx = drift_idx[mask[drift_idx]]
        drift_times = timestamps.iloc[drift_idx]
        for ax in axes:
            for drift_time in drift_times:
                ax.axvline(drift_time, color='tab:purple', linestyle='--', lw=1.0, alpha=0.7, label='Drift Points')

    axes[-1].set_xlabel(timestamp_col)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig, axes


def plot_membership_sel(
    clf,
    df_data=None,
    data=None,
    label=None,
    timestamp_col='timestamp',
    traffic_choose='total_flow',
    drift_points = None,
    stride=24,
    start_date=None,
    end_date=None,
    figsize=None,
    delay=0,
    sel_mem=True,
    cols=None,
):
    mems = clf.mem_nm if sel_mem else clf.dist_nm
    mem_n = max(len(mem) for mem in mems)

    dist_ms = clf.dist_nm
    dist_n = max(len(dist) for dist in dist_ms)

    if cols is None:
        cols = list(range(len(mems)))
    else:
        cols = list(cols)

    invalid_cols = [col for col in cols if col < 0 or col >= len(mems)]
    if invalid_cols:
        raise IndexError(f'cols out of range: {invalid_cols}')

    # drift_points = getattr(clf, 'drift_points', None)
    colors = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
    ]
    markers = ['o', 's', '^', 'P', 'v', 'D', 'X', '*', '<', '>']
    linestyles = ['-', ':', '--', '-.', '-', '-.', '--', ':']

    if df_data is None:
        if figsize is None:
            figsize = (14, 2)
        plt.figure(figsize=figsize)
        for i, col in enumerate(cols):
            mem2 = np.asarray(mems[col], dtype=float)[delay:]
            if len(mem2) < mem_n:
                mem2 = np.pad(mem2, (mem_n - len(mem2), 0), 'constant', constant_values=0)
            else:
                mem2 = mem2[-mem_n:]
            plt.plot(
                mem2,
                linestyle='-',
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=f'mem {i}',
            )
        if drift_points is not None and len(drift_points) > 0:
            drift_positions = np.asarray(drift_points, dtype=float) / stride
            drift_positions = drift_positions[(0 <= drift_positions) & (drift_positions < mem_n)]
            for drift_pos in drift_positions:
                plt.axvline(drift_pos, color='tab:purple', linestyle='--', lw=1.0, alpha=0.7)
        plt.legend()
        plt.show()
        return

    df_plot = df_data.copy()
    df_plot[timestamp_col] = pd.to_datetime(df_plot[timestamp_col])
    timestamps = df_plot[timestamp_col]

    if data is None:
        data = df_plot[traffic_choose].to_numpy()
    data = np.asarray(data)

    mask = np.ones(len(df_plot), dtype=bool)
    if start_date is not None:
        mask &= timestamps >= pd.to_datetime(start_date)
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        if isinstance(end_date, str) and len(end_date.strip()) <= 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        mask &= timestamps <= end_ts
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        raise ValueError(f'No data in selected range: {start_date} ~ {end_date}')

    mem_idx = np.arange(mem_n) * stride
    mem_idx = mem_idx[mem_idx < len(df_plot)]
    mem_time = timestamps.iloc[mem_idx]
    mem_mask = mask[mem_idx]

    if figsize is None:
        figsize = (14, 6)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={'height_ratios': [1.2, 1, 1.2]},
    )
    nan_to_int_times = []

    axes[0].plot(timestamps[mask], data[mask], 'k-', lw=1.2, label='Traffic Data')
    if label is not None:
        label = np.asarray(label)
        anom_mask = mask & (label == 1)
        axes[0].scatter(timestamps[anom_mask], data[anom_mask], color='red', s=18, zorder=3, label='Anomaly')
    axes[0].tick_params(axis='y', labelrotation=90)
    axes[0].set_ylabel('Traffic (flow)', fontsize=22)
    
    axes[0].grid(alpha=0.25)

    for i, col in enumerate(cols):
        mem2 = np.asarray(mems[col], dtype=float)[delay:]
        if len(mem2) < mem_n:
            mem2 = np.pad(mem2, (mem_n - len(mem2), 0), 'constant', constant_values=np.nan)
        else:
            mem2 = mem2[-mem_n:]
        mem2 = mem2[:len(mem_idx)]
        if i == len(cols) - 1:
            nan_to_int_idx = np.flatnonzero(np.isnan(mem2[:-1]) & ~np.isnan(mem2[1:])) + 1
            nan_to_int_idx = nan_to_int_idx[nan_to_int_idx < len(mem_mask)]
            nan_to_int_idx = nan_to_int_idx[mem_mask[nan_to_int_idx]]
            nan_to_int_times = list(mem_time.iloc[nan_to_int_idx])
        axes[1].plot(
            mem_time[mem_mask],
            mem2[mem_mask],
            # linestyle='-',
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle = linestyles[i% len(linestyles)],
            lw=2,
            ms=7,
            label=rf'$N^M_{{{i}}}$',
        )
        dist2 = np.asarray(dist_ms[col], dtype=float)[delay:]
        if len(dist2) < dist_n:
            dist2 = np.pad(dist2, (dist_n - len(dist2), 0), 'constant', constant_values=np.nan)
        else:
            dist2 = dist2[-dist_n:]
        dist2 = dist2[:len(mem_idx)]
        # print('dist:', len(mem2), len(dist2)
        axes[2].plot(
            mem_time[mem_mask],
            dist2[mem_mask]/10000,
            # linestyle='-',
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linestyle = linestyles[i% len(linestyles)],
            lw=2,
            ms=7,
            label=f'dist {i}',
        )

    axes[1].set_ylabel((r'$\phi$'), fontsize=23)

    axes[2].set_ylabel('Normalized \nDistance', fontsize=22)
    # axes[2].legend(loc='upper right', ncol=min(len(cols), 4))
    axes[2].grid(alpha=0.25)

    for ax in axes[1:3]:
        y_min, y_max = ax.get_ylim()
        y_span = y_max - y_min
        arrow_tip_y = y_max - 0.08 * y_span
        arrow_start_y = y_max + 0.18 * y_span
        for event_time in nan_to_int_times:
            ax.axvline(event_time, color='black', linestyle='--', lw=2.5, alpha=0.9)
            ax.annotate(
                '',
                xy=(event_time, arrow_tip_y),
                xytext=(event_time, arrow_start_y),
                arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=5.0),
                annotation_clip=False,
            )

    axes[1].axvline(event_time, color='black', linestyle='--', lw=2.5, alpha=0.9, label=r'Add $N^M_{3}$')

    axes[1].legend(
        ncol=6, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 3.035), 
        frameon=False, 
        fontsize=23,
        columnspacing=1.5,
        handletextpad=1
        )
    axes[1].grid(alpha=0.25)



    if drift_points is not None and len(drift_points) > 0:
        # drift_idx = np.asarray(drift_points, dtype=int)
        # drift_idx = drift_idx[(0 <= drift_idx) & (drift_idx < len(df_plot))]
        # drift_idx = drift_idx[mask[drift_idx]]
        # drift_times = timestamps.iloc[drift_idx]
        drift_times = [pd.to_datetime(f) for f in drift_points]
        drift_times = [f for f in drift_times if f > pd.to_datetime(start_date) and f < pd.to_datetime(end_date)]
        print(drift_times)
        for ax in axes:
            for drift_time in drift_times:
                ax.axvline(drift_time, color='tab:gray', linestyle='--', lw=1.5, alpha=1)

        axes[0].axvline(drift_time, color='tab:gray', linestyle='--', lw=1.5, alpha=1, label='Drift Points')

    axes[0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.8), frameon=False, fontsize=23)
    # axes[-1].set_xlabel(timestamp_col)
    date_locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[-1].xaxis.set_major_locator(date_locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    axes[-1].tick_params(axis='x', labelrotation=0)
    axes[-1].xaxis.get_offset_text().set_fontsize(20)
    axes[-1].set_xlabel('Time', fontsize=23)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=20)
    for ax in axes:
        ax.margins(x=0)
    # fig.autofmt_xdate(rotation=30)
    # fig.tight_layout()
    return fig, axes
