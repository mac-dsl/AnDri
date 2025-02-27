from util.create_drift import get_arff_data_labels, find_anomaly_intervals
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys


COLOURS = {
    0: 'tab:blue',
    1: 'tab:green',
    2: 'tab:red',
    3: 'tab:cyan',
    4: 'tab:pink',
    5: 'tab:purple',
    'drift': 'gold'
}


class PlotStream:

    #  @param source_dir: string, filepath to directory containing source files
    #  @param drift_dir: string, filepath to directory containinng output files
    #  @param filename: string, name of file corresponding to drift stream
    #  @param colours: dictionary representing background colours for
    #        denoting stream segments, default COLOURS
    def __init__(self, source_dir, drift_dir, filename, colours=COLOURS):
        """
        Constructor method for plotting generated drift streams
        """
        drift_path = f'{drift_dir}/{filename}.arff'
        X, y = get_arff_data_labels(drift_path)
        params = self.get_plot_params(
            drift_dir, filename
        )
        self.drift_name = filename          # string, file name
        self.drift_X = X                    # ndarray, data (drift stream)
        self.drift_y_anom = y               # ndarray, anomaly labels (drift stream)
        self.length = len(y)                # int, length of drift stream
        self.drift_label = params[0]        # list of int, labels indicating drift, {0,1}
        self.source_streams = params[1]     # list of string, source file names
        self.streams = params[2]            # list of int, stream order
        self.positions = params[3]          # list of int, center positions for drift
        self.w_drift = params[4]         # list of boolean, drift inserted before anomaly
        self.seq_before = params[5]         # list of boolean, drift inserted before anomaly
        Xs, ys = self.get_source_data_labels(source_dir)
        self.source_Xs = Xs                 # list of ndarray, data
        self.source_ys = ys                 # list of ndarray, anomaly labels
        self.colours = colours

    #  @params source_dir: string, filepath to directory containing source files
    #  @param filename: string, name of files corresponding to drift stream
    #  @Return drift_label, source_streams, streams, positions
    def get_plot_params(self, source_dir, filename):
        """
        Retrieve parameters required for plotting generated stream
        """
        filepath = f'{source_dir}/{filename}.arff'
        drift_label_path = f'{source_dir}/{filename}.csv'

        # Collect drift label from CSV
        drift_label = pd.read_csv(drift_label_path)
        drift_label = drift_label['0'].tolist()

        # Collect stream construction info from ARFF header
        with open(filepath, 'r') as f:
            stream_info = []
            for line in f:
                if line[0] == '%':
                    stream_info.append(line)
                else:
                    break
        source_streams = []
        for line in stream_info:
            line = line.strip('% ')
            if line.split(':')[0] == 'Stream Order':
                streams = line.split(': ')[-1]
                streams = ast.literal_eval(streams)
            elif line.split(':')[0] == 'Drift Positions':
                positions = line.split(': ')[-1]
                positions = ast.literal_eval(positions)
                positions = [0] + positions
            elif line.split(':')[0] == 'Drift Widths':
                w_drift = line.split(': ')[-1]
                w_drift = ast.literal_eval(w_drift)
            elif line.split(':')[0] == 'Drift Before':
                seq_before = line.split(': ')[-1]
                seq_before = ast.literal_eval(seq_before)
            else:
                if line.split(':')[0] != 'Source Streams':
                    source_streams.append(line.split(':')[-1].strip())

        #  @Returns
        #    drift_label: list of int, whether or not a point is labelled as drift
        #    source_streams: list of string, source file names
        #    streams: list of int, denoting order of streams in combination
        #    positions: list of int, center position of drift
                    
        return drift_label, source_streams, streams, positions, w_drift, seq_before

    #  @params source_dir: string, filepath to directory containing source files
    #  @Return Xs: list of ndarray, contains data for source streams
    #  @Return ys: list of ndarray, contains anomaly labels for source streams
    def get_source_data_labels(self, source_dir):
        """
        Retrieve data and labels for source streams
        """
        Xs, ys = [], []
        for s in self.source_streams:
            X, y = get_arff_data_labels(os.path.join(source_dir, s))
            Xs.append(X)
            ys.append(y)
        return Xs, ys

    #  @params X: int iterable, data points
    #  @params y: int iterable, anomaly labels
    #  @params ax: Axes object to plot graph
    #  @params start: int, start of interval to plot
    #  @params end: int, endn of interval to plot
    #  @params title: string, title of plot
    #  @params marker: string, marker type for plot, default '-' (line)
    #  @params size: int, fontsize, default=10
    def plot_anomaly(
            self, X, y, ax, start=0, end=sys.maxsize, title="", marker="-", size=10, show_label=False
    ):
        """
        Plot the data with highlighted anomaly
        """
        start = max(0, start)
        end = min(X.shape[0], end)
        ax.plot(np.arange(start, end), X[start:end],
                f"{marker}b", label="_"*(not show_label) + "Non-Anomalous")
        anom_ints = [
            ai for ai in find_anomaly_intervals(y) if
            (ai[0] < end and ai[0] > start) or
            (ai[1] < end and ai[1] > start) or
            (ai[0] < start and ai[1] > end)
        ]
        for (i, (anom_start, anom_end)) in enumerate(anom_ints):
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            label = "_"*(i + (not show_label)) + "Anomaly"
            ax.plot(
                np.arange(anom_start, anom_end),
                X[anom_start:anom_end],
                f"{marker}r",
                label=label
            )
        if len(title) > 0:
            ax.set_title(title, size=size)

    #  @param positions: list of int, center position of drift
    #  @params y: int iterable, drift labels
    #  @params streams: list of int, denoting order of streams in combination
    #  @params ax: Axes object to plot graph
    #  @params colours: dictionary representing background colours for
    #        denoting stream segments
    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def plot_stream_drift(
        self, positions, y, streams, ax, colours=COLOURS, start=0, end=sys.maxsize, show_source_labels=True
    ):
        """
        Plot source stream and drift periods via background colour
        """
        positions = positions + [len(y)]
        if show_source_labels:
            show_label_streams = {j: 0 for j in range(len(streams))}
        else:
            show_label_streams = {j: 1 for j in range(len(streams))}
        for i in range(0, len(positions)-1):
            if positions[i] > end:
                break
            elif positions[i+1] < start:
                pass
            else:
                s_start = max(start, positions[i])
                s_end = min(positions[i+1], end)
                colour = colours[streams[i]]
                s = streams[i]
                label = '_'*show_label_streams[s] + f'stream {s}'
                ax.axvspan(
                    s_start, s_end, facecolor=colour, alpha=0.3, label=label
                )
                show_label_streams[s] += 1
        drift_ints = [
            di for di in find_anomaly_intervals(y) if
            (di[0] < end and di[0] > start) or
            (di[1] < end and di[1] > start) or
            (di[0] < start and di[1] > end)
        ]
        show_label = 0
        for (i, (drift_start, drift_end)) in enumerate(drift_ints):
            drift_start = max(start, drift_start)
            drift_end = min(end, drift_end)
            label = '_'*show_label + 'drift'
            ax.axvspan(
                drift_start, drift_end, facecolor=colours['drift'], alpha=1, label=label
            )
            show_label += 1

    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def plot_all(self, start=0, end=sys.maxsize):
        """
        Plot all source streams and drift stream between an interval
        """
        end = min(end, self.length)
        start = max(0, start)
        rows = len(self.source_streams) + 1
        fig, ax = plt.subplots(rows, 1, figsize=(10, 2.5*rows), sharey=True, sharex=True)

        # for each source stream, plot stream and background colour
        for (i, s) in enumerate(self.source_streams):
            X, y = self.source_Xs[i], self.source_ys[i]
            colour = self.colours[i]
            self.plot_anomaly(
                X, y, ax[i], start, end, title=s, size=8, show_label=(i == 0)
            )
            ax[i].axvspan(
                start, end, facecolor=colour, alpha=0.3, label=f'stream {i}'
            )
            ax[i].tick_params(axis='x', labelsize=8)

        # Plot drift stream
        self.plot_anomaly(
            self.drift_X, self.drift_y_anom, ax[-1], start, end, self.drift_name, size=8
        )
        self.plot_stream_drift(
            positions=self.positions,
            y=self.drift_label,
            streams=self.streams,
            ax=ax[-1],
            colours=self.colours,
            start=start,
            end=end,
            show_source_labels=False
        )
        fig.legend(loc='lower right')

    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def plot_drift(self, start=0, end=sys.maxsize):
        """
        Plot only the drift stream between an interval
        """
        end = min(end, self.length)
        fig, ax = plt.subplots(figsize=(10, 3))
        self.plot_anomaly(
            self.drift_X, self.drift_y_anom, ax, start, end, self.drift_name, size=8
        )
        self.plot_stream_drift(
            self.positions, self.drift_label, self.streams, ax, self.colours, start, end
        )
        fig.legend(loc='lower right')

    #  @param k: int, to indicate the kth drift to plot
    #             value range between 1 and n_drift
    #  @param w: int, half width of interval to view
    def plot_anomaly_k(self, k, w=1000):
        """
        Plot only drift surrounding the kth anomaly
        """
        start = self.positions[k] - w
        end = self.positions[k] + w
        start = max(0, start)
        end = min(end, self.length)
        self.plot_drift(start, end)

    #  @param k: int, to indicate the kth drift to plot
    #             value range between 1 and n_drift
    #  @param w: int, half width of interval to view
    def plot_anomaly_k_with_source(self, k, w=1000):
        """
        Plot drift and parent streams surrounding the kth
        anomaly
        """
        start = self.positions[k] - w
        end = self.positions[k] + w
        start = max(0, start)
        end = min(end, self.length)
        parent_i = [self.streams[k-1], self.streams[k]]
        parent_streams = [self.source_streams[i] for i in parent_i]
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

        # for each source stream, plot stream and background colour
        for (i, s) in enumerate(parent_streams):
            X, y = self.source_Xs[parent_i[i]], self.source_ys[parent_i[i]]
            colour = COLOURS[parent_i[i]]
            self.plot_anomaly(X, y, ax[i], start, end, title=s, size=8, show_label=False)
            ax[i].axvspan(start, end, facecolor=colour, alpha=0.3, label=f'stream {parent_i[i]}')
            ax[i].tick_params(axis='x', labelsize=8)

        # Plot drift stream
        self.plot_anomaly(
            self.drift_X, self.drift_y_anom, ax[-1], start, end, self.drift_name, size=8,
        )
        self.plot_stream_drift(
            positions=self.positions,
            y=self.drift_label,
            streams=self.streams,
            ax=ax[-1],
            colours=self.colours,
            start=start,
            end=end,
            show_source_labels=False
        )
        fig.legend(loc='lower right')
