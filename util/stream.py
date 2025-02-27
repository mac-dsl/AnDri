import arff
import ast
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import sys
from matplotlib.lines import Line2D
from scipy import signal


COLOURS = {
    0: 'tab:blue',
    1: 'tab:green',
    2: 'tab:red',
    3: 'tab:cyan',
    4: 'tab:pink',
    5: 'tab:purple',
    'drift': 'gold'
}



class Stream:
    """
    Class representing a stream of data
    """
    def __init__(self, filepath) -> None:
        filename = filepath.split("/")[-1]
        self.path = "/".join(filepath.split("/")[:-1])
        self.filename = ".".join(filename.split(".")[:-1])
        self.data = None
        self.anomaly_labels = None
        self.anomaly_intervals = None

        file_ext = filename.split(".")[-1].lower()
        if file_ext in {"csv", "out"}:
            self.data, self.anomaly_labels = self.__get_csv_data_labels(filepath)
            self.drift_labels = self.__get_drift_labels()
            self.length = len(self.anomaly_labels)
        elif file_ext == "arff":
            self.data, self.anomaly_labels = self.__get_arff_data_labels(filepath)
            self.drift_labels = self.__get_drift_labels()
            self.length = len(self.anomaly_labels)


    #  @params dir: str, path to directory to write arff file to
    def to_arff(self, dir=".", start=0, end=sys.maxsize):
        """
        Method to export Stream object to arff file in specified directory
        """

        data_labels = np.concatenate([self.data[max(0,start):min(end,self.data.size)], self.anomaly_labels[max(0,start):min(end,self.data.size)]], axis=1)
        content = pd.DataFrame(data_labels)
        content = content.to_csv(header=False, index=False).strip("\n").split("\n")
        content = [f"{line},\n" for line in content]
        header = f"@relation '{dir}/{self.filename}'\n\n"
        header += "@attribute att1 numeric\n"
        header += "@attribute class {1.0, 0.0}\n\n"
        header += "@data\n\n"
        # arff_filename = f"{self.filename}.arff"
        arff_filename = f"{dir}/{self.filename}.arff"
        with open(arff_filename, "w") as output_file:
            output_file.writelines([header] + content)
        return arff_filename


    def get_anomaly_intervals(self):
        """
        Return the intervals where there is an anomaly in the 
        form of a list of tuples formatted as (start, end)
        """
        if self.anomaly_intervals is None:
            self.__set_anomaly_intervals()
        return self.anomaly_intervals


    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def plot(self, start=0, end=sys.maxsize) -> None:
        end = min(end, self.length)
        plt.rcParams['figure.dpi'] = 300
        fig, ax = plt.subplots(figsize=(10, 3))
        # may need to remove later if necessary
        self.__set_anomaly_intervals()
        self.plot_anomaly(
            ax, start, end, self.filename, size=8, show_label=True
        )
        non_anomalous_line = Line2D([0], [0], color='blue', lw=2)
        anomalous_line = Line2D([0], [0], color='red', lw=2)
        fig.legend([non_anomalous_line, anomalous_line], ['Non-Anomalous', 'Anomalous'])
        fig.legend(loc='upper right')
        


    #  @param k: int, to indicate the kth anomaly to plot
    #             value range between 1 and n_anomalies
    #  @param w: int, half width of interval to view
    def plot_anomaly_k(self, k, w=1000) -> None:
        """
        Plot only drift surrounding the kth anomaly
        """
        anom_center = sum(self.anomaly_intervals[k-1]) // 2
        start = anom_center - w
        end = anom_center + w
        start = max(0, start)
        end = min(end, self.length)
        self.plot(start, end)


    #  @params ax: Axes object to plot graph
    #  @params start: int, start of interval to plot
    #  @params end: int, endn of interval to plot
    #  @params title: string, title of plot
    #  @params marker: string, marker type for plot, default '-' (line)
    #  @params size: int, fontsize, default=10
    def plot_anomaly(
            self, ax, start=0, end=sys.maxsize, title="", marker="-", size=10, show_label=False
    ):
        """
        Plot the data with highlighted anomaly
        """
        start = max(0, start)
        end = min(self.data.shape[0], end)
        ax.plot(np.arange(start, end), self.data[start:end],
                f"{marker}b", label="_"*(not show_label) + "Non-Anomalous")
       
        self.__set_anomaly_intervals()
        anom_ints = [
            ai for ai in self.anomaly_intervals if
            (ai[0] < end and ai[0] > start) or
            (ai[1] < end and ai[1] > start) or
            (ai[0] < start and ai[1] > end)
        ]
        for (i, (anom_start, anom_end)) in enumerate(anom_ints):
            anom_start = max(start, anom_start)
            anom_end = min(end, anom_end)
            label = "_"*(i + (not show_label)) + "Anomaly"
            # TO GO BACK TO REGULAR LINEWIDTH, REMOVE LINEWIDTH PARAMETER
            ax.plot(
               np.arange(anom_start, min(len(self.data),anom_end+1)),
                self.data[anom_start: min(len(self.data),anom_end+1)],
                # f"{marker}r", linewidth=3
                f"{marker}r", linewidth=2
            )
        if len(title) > 0:
            ax.set_title(title, size=size)



    def __set_anomaly_intervals(self):
        """
        Method to find the intervals where there is an anomaly
        """
        if sum(self.anomaly_labels) > 0:
            change_indices = np.where(np.diff(self.anomaly_labels, axis=0) != 0)[0]
            if len(change_indices) == 0:
                return []
            anom_intervals = []

            if self.anomaly_labels[change_indices[0]] == 0:
                i = 0
            else:
                i = 1
                anom_intervals.append([0, change_indices[0]+1])

            while i + 1 < len(change_indices):
                anom_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
                i += 2

            if self.anomaly_labels[-1] == 1:
                anom_intervals.append([change_indices[-1]+1, len(self.anomaly_labels)])

            self.anomaly_intervals = anom_intervals
        
        else:
            self.anomaly_intervals = []


    #  @params filename : str, filename of csv or out data source
    #  @Return tuple of ndarrays, ndarrays corresponding to data (N, 1) and labels 
    #     (N,) from arff data
    def __get_csv_data_labels(self, filename: str):
        """
        Find ndarray corresponding to data and labels from csv formatted data
        """
        csv_content = pd.read_csv(filename, header=None)
        csv_len = csv_content.shape[0]
        data = csv_content[0].to_numpy().reshape(csv_len, 1)
        anomaly_labels = csv_content[1].to_numpy().reshape(csv_len, 1)
        return data.astype(float), anomaly_labels.astype(float)


    #  @params filename : str, filename of arff data source
    #  @Return tuple of ndarrays, ndarrays corresponding to data (N, 1) and labels 
    #     (N,) from arff data
    def __get_arff_data_labels(self, filename: str):
        """
        Find ndarray corresponding to data and labels from arff data
        """
        arff_content = arff.load(f.replace(',\n', '\n') for f in open(filename, 'r'))
        arff_data = arff_content['data']
        data = np.array([i[:1] for i in arff_data])
        anomaly_labels = np.array([i[-1] for i in arff_data])
        anomaly_labels = anomaly_labels.reshape((len(anomaly_labels),1))
        return data.astype(float), anomaly_labels.astype(float)


    def __get_drift_labels(self):
        return np.zeros((len(self.anomaly_labels),1))

# function to transform stream, to create a new stream
def transform_stream(stream: Stream, start: float, end: float, drift_scale: float) -> Stream:
    transformed_stream = stream
    dataset = transformed_stream.data
    if start < 1:
        cd1 = round(len(dataset)*start)
        if end < 1:
            cd2 = round(len(dataset)*end)
        if start > 1:
            cd1 = start
        if end > 1:
            cd2 = end
    labels = transformed_stream.anomaly_labels
    # in here, ratio = drift_scale (if ratio < 1 -> inc. freq.)
    val=drift_scale
    d_temp = dataset[cd1:cd2]
    wid_len = int((cd2-cd1)*val)
    d_mod = signal.resample(d_temp, wid_len) 
    l_temp = labels[cd1:cd2]
    l_mod = signal.resample(l_temp, wid_len)
    l_mod = np.round(l_mod) 
    transformed_stream.anomaly_labels = np.concatenate((labels[:cd1], l_mod, labels[cd2:]))
    transformed_stream.anomaly_labels = np.absolute(transformed_stream.anomaly_labels)
    # caution! the length of the stream data attribute is changed here
    transformed_stream.data = np.concatenate((dataset[:cd1], d_mod, dataset[cd2:]))
    transformed_stream.drift_labels = np.zeros((len(transformed_stream.anomaly_labels),1))
    transformed_stream.length = len(transformed_stream.anomaly_labels)
    return transformed_stream


class DriftStream(Stream):
    """
    Stream class representing data with generated drift
    """
            
    def __init__(self, filepath, source_dir, colours=COLOURS):
        super().__init__(filepath)
        drift_params = self.__get_drift_params()
        drift_labels = np.array(drift_params[0])
        self.drift_labels = drift_labels.reshape(len(drift_labels),1)
        self.source_streams = []
        source_streams = drift_params[1]
        for s in source_streams:
            try:
                self.source_streams.append(Stream(f"{source_dir}/{s}"))
            except:
                null_stream = Stream("")
                null_stream.data = np.zeros(self.length)
                null_stream.anomaly_labels = np.zeros(self.length)
                null_stream.drift_labels = np.zeros(self.length)
                self.source_streams.append(null_stream)
        self.streams = drift_params[2]
        self.positions = drift_params[3]
        self.w_drift = drift_params[4]
        self.seq_before = drift_params[5]
        self.drift_intervals = None
        self.colours = colours


    #  @params dir: str, path to directory to write arff file to
    def to_arff(self, dir=".", start=0):
        """
        Method to export Stream object to arff file in specified directory
        """
        arff_filename = f"{self.filename}.arff"
        arff_content = open(f"{dir}/{self.filename}.arff", "r")
        with open(arff_filename, "w") as output_file:
            output_file.writelines(arff_content)
        return arff_filename


    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, end of interval to plot, default: sys.maxsize 
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
            colour = self.colours[i]
            s.plot_anomaly(
                ax[i], start, end, title=s.filename, size=8, show_label=(i == 0)
            )
            ax[i].axvspan(
                start, end, facecolor=colour, alpha=0.3, label=f'stream {i}'
            )
            ax[i].tick_params(axis='x', labelsize=8)

        # Plot drift stream
        super().plot_anomaly(
            ax[-1], start, end, self.filename, size=8
        )
        self.__plot_stream_drift(
            ax=ax[-1],
            colours=self.colours,
            start=start,
            end=end,
            show_source_labels=False
        )
        fig.legend(loc='lower right')


    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, end of interval to plot, default: sys.maxsize 
    #  @params title: str, title of plot
    def plot_drift(self, start=0, end=sys.maxsize, title=None):
        """
        Plot only the drift stream between an interval
        """
        end = min(end, self.length)
        fig, ax = plt.subplots(figsize=(10, 3))
        if title is None:
            title = self.filename
        super().plot_anomaly(
            ax, start, end, title, size=8
        )
        self.__plot_stream_drift(
            ax, self.colours, start, end
        )
        fig.legend(loc='lower right')


    #  @param k: int, to indicate the kth drift to plot
    #             value range between 1 and n_drift
    #  @param w: int, half width of interval to view
    def plot_drift_k(self, k, w=1000):
        """
        Plot only drift surrounding the kth anomaly
        """
        start = self.positions[k] - w
        end = self.positions[k] + w
        start = max(0, start)
        end = min(end, self.length)
        before = self.seq_before[k-1]
        title = f"{self.filename}, k = {k}, Before = {before}"
        self.plot_drift(start, end, title)


    #  @param k: int, to indicate the kth drift to plot
    #             value range between 1 and n_drift
    #  @param w: int, half width of interval to view
    def plot_drift_k_with_source(self, k, w=1000):
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
            colour = COLOURS[parent_i[i]]
            s.plot_anomaly(ax[i], start, end, title=s.filename, size=8, show_label=False)
            ax[i].axvspan(start, end, facecolor=colour, alpha=0.3, label=f'stream {parent_i[i]}')
            ax[i].tick_params(axis='x', labelsize=8)

        before = self.seq_before[k-1]
        title = f"{self.filename}, k = {k}, Before = {before}"

        # Plot drift stream
        super().plot_anomaly(
            ax[-1], start, end, title, size=8,
        )
        self.__plot_stream_drift(
            ax=ax[-1],
            colours=self.colours,
            start=start,
            end=end,
            show_source_labels=False
        )
        fig.legend(loc='lower right')


    #  @params ax: Axes object to plot graph
    #  @params colours: dictionary representing background colours for
    #        denoting stream segments
    #  @params start: int, start of interval to plot, default: 0
    #  @params end: int, endn of interval to plot, default: sys.maxsize 
    def __plot_stream_drift(
        self, ax, colours=COLOURS, start=0, end=sys.maxsize, show_source_labels=True
    ):
        """
        Plot source stream and drift periods via background colour
        """
        positions = self.positions + [len(self.anomaly_labels)]
        if show_source_labels:
            show_label_streams = {j: 0 for j in range(len(self.streams))}
        else:
            show_label_streams = {j: 1 for j in range(len(self.streams))}
        for i in range(0, len(positions)-1):
            if positions[i] > end:
                break
            elif positions[i+1] < start:
                pass
            else:
                s_start = max(start, positions[i])
                s_end = min(positions[i+1], end)
                colour = colours[self.streams[i]]
                s = self.streams[i]
                label = '_'*show_label_streams[s] + f'stream {s}'
                ax.axvspan(
                    s_start, s_end, facecolor=colour, alpha=0.3, label=label
                )
                show_label_streams[s] += 1
        if self.drift_intervals is None:
            self.__set_drift_intervals()
        drift_ints = [
            di for di in self.drift_intervals if
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


    def __set_drift_intervals(self):
        """
        Method to find the intervals where there is an drift
        """
        change_indices = np.where(np.diff(self.drift_labels, axis=0) != 0)[0]
        if len(change_indices) == 0:
            return []
        drift_intervals = []

        if self.drift_labels[change_indices[0]] == 0:
            i = 0
        else:
            i = 1
            drift_intervals.append([0, change_indices[0]+1])

        while i + 1 < len(change_indices):
            drift_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
            i += 2

        if self.drift_labels[-1] == 1:
            drift_intervals.append([change_indices[-1]+1, len(self.drift_labels)])

        if drift_intervals is None:
            drift_intervals = []

        self.drift_intervals = drift_intervals


    #  @Returns
    #    drift_label: list of int, whether or not a point is labelled as drift
    #    source_streams: list of string, source file names
    #    streams: list of int, denoting order of streams in combination
    #    positions: list of int, center position of drift
    def __get_drift_params(self):
        """
        Retrieve parameters required for plotting generated stream
        """
        filepath = f'{self.path}/{self.filename}.arff'
        drift_label_path = f'{self.path}/{self.filename}.csv'

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
                    
        return drift_label, source_streams, streams, positions, w_drift, seq_before
