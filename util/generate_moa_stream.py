from util.plot_stream import get_arff_data_labels
from util.create_drift import get_split_index, get_split_index_uniform, get_stream_cuts
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys


class GenMOAStream:

    #  @param source_dir: string, filepath to directory containing source files
    #  @param num_streams: innt, number of files to consider in combination
    #  @param drift_dir: string, filepath to directory containinng output files
    #  @param moa_path: string, filepath to MOA executable
    #  @attr selected_streams: list of string, arff filenames of source streams
    #  @attr max_stream: int, max stream index (ex. for 6 streams, index 5 is max)
    def __init__(
            self, source_dir, drift_dir, moa_path, num_streams=6, selected_streams=None
    ):
        """
        Constructor method for drift generation object
        """
        self.source_dir = source_dir
        self.drift_dir = drift_dir
        self.moa_path = moa_path

        if selected_streams is None:
            # Randomly select files to be used to create drift stream
            files = os.listdir(source_dir)
            files = [f for f in files if f.split('.')[-1] == 'arff']
            selected_streams = np.random.choice(files, (num_streams,), replace=False)
            self.selected_streams = selected_streams.tolist()
            self.max_stream = len(self.selected_streams) - 1
        else:
            self.selected_streams = selected_streams
            self.max_stream = len(selected_streams) - 1

        # Locate anomalies for all streams and record them within a
        # master list
        anom_ints = []
        for s in selected_streams:
            _, y = get_arff_data_labels(os.path.join(self.source_dir, s))
            anom_ints.append(self.find_anomaly_intervals(y))
        self.total_anom_ints = self.get_total_anoms(anom_ints)

    #  @Return df: pandas dataframe, containing descriptive information on the
    #       source data (ie. number of anomalies, average anomaly length, etc)
    def get_source_summary(self):
        df = pd.read_csv(f"{self.source_dir}/description.csv")
        df = df.loc[df.filename.apply(lambda x: x in self.selected_streams)]
        return df

    #  @param length: int, total length of new stream
    #  @param p_drift: float, target percent of data points classified as drift
    #  @param n_drift: int, target number of drift sequences
    #  @param p_before: float, target percent of drift coming before anomaly
    #  @param sub_dir: string, name of subdirectory to export drift stream
    #  @param dataset: string, descriptor (name) or source dataset for identification
    #  @param mode: int, indicator for drift assembly method, options {0,1}, default 0
    #           Mode 0: variable drift widths and positions
    #           Mode 1: uniform drift widths and positions (helpful for high p_drift)
    #  @Returns output_path, drift_label, positions, streams, seq_before
    def run_generate_grad_stream_moa(
        self, length, p_drift, n_drift, p_before, sub_dir, dataset, mode=0
    ):
        """
        Create new gradual drift-injected stream using MOA based on parameters
        """
        print('Generating splits...')
        if mode == 0:
            streams, positions, w_drift, seq_before = \
                get_split_index(
                    length, p_drift, n_drift, p_before, self.max_stream, self.total_anom_ints
                )
        else:
            streams, positions, w_drift, seq_before =\
                get_split_index_uniform(
                    length, p_drift, n_drift, p_before, self.max_stream, self.total_anom_ints
                )
        print('Done!')

        output_path, drift_label = self.assemble_drift_stream(
            positions, streams, w_drift, seq_before, sub_dir, length, dataset
        )

        #  @Returns
        #    output_path: string, path to file where generated data stream is exported to
        #    drift_label: list of int, whether or not a point is labelled as drift
        #    streams: list of int, denoting order of streams in combination
        #    positions: list of int, center position of drift
        #    seq_before: list of boolean indicating whether drift comes before or before anomaly

        return output_path, drift_label, streams, positions, seq_before, w_drift

    #  @param positions: list of int, center position of drift
    #  @param streams: list of int, denoting order of streams in combination
    #  @param w_drift: int, width of drift
    #  @param seq_before: list of boolean indicating whether drift comes before or before anomaly
    #  @param sub_dir: string, name of subdirectory to export drift stream
    #  @param length: int, total length of new stream
    #  @param dataset: string, descriptor (name) or source dataset for identification
    #  @Returns output_path, drift_label
    def assemble_drift_stream(
            self, positions, streams, w_drift, seq_before, sub_dir, length, dataset
    ):
        """
        Function to combine helper methods to create drift stream
        """
        print('Getting stream file cuts...', end='\t')
        stream_cuts = get_stream_cuts(positions, seq_before, w_drift, streams, self.max_stream)
        print('Done!')

        print('Creating intermediate files...', end='\t')
        streams_intermed = self.create_intermediate_files(stream_cuts)
        print('Done!')

        print('Recursively generating MOA command...', end='\t')
        drift_stream = self.generate_drift_stream_for_moa(
            streams, positions, w_drift, streams_intermed
        )
        print('Done!')

        output_path, filename = self.get_output_filepath(
            w_drift, length, positions, seq_before, dataset, sub_dir
        )
        print('Drift filename: ', filename)

        print('Running terminal command...', end='\t')
        self.run_moa_command(drift_stream, length, output_path)
        # Add information on stream construction to ARFF file as comments
        self.add_stream_info(
            output_path,
            self.selected_streams,
            streams,
            positions,
            w_drift,
            seq_before
        )
        print('Done!')

        print('Generating drift labels...', end='\t')
        drift_label = self.get_drift_labels(positions, w_drift, length)
        drift_label_df = pd.DataFrame(drift_label)
        drift_label_df.to_csv(f'{self.drift_dir}/{sub_dir}/{filename}.csv')
        print('Done!')

        #  @Returns
        #    output_path: string, path to file where generated data stream is exported to
        #    drift_label: list of int, whether or not a point is labelled as drift

        return output_path, drift_label
    
    #  @param stream_cuts: list of list of int, indices to split intermediate
    #                 ARFF files in the form [L_0, L_1, ... , L_{N-1}], where
    #                 each L_i is a list of int
    #  @Returns streams_intermed: list of string, list of cut stream segments
    #                          in order of assembly to generate stream
    def create_intermediate_files(self, stream_cuts):
        """
        Create intermediate files from source based on given indices to cut
        """
        streams_intermed = []
        for i in range(len(self.selected_streams)):
            f = os.path.join(self.source_dir, self.selected_streams[i])
            split_index = stream_cuts[i]
            int_dir = f"{self.drift_dir}/intermediate"
            int_name = self.split_arff(f, split_index, 'intermed', int_dir)
            streams_intermed.append(int_name)
        return streams_intermed

    #  @param drift_stream: list of string, list of cut stream segments
    #                          in order of assembly to generate stream
    #  @param length: int, total length of new stream
    #  @param output_path: string, path to file where generated data stream is exported to
    def run_moa_command(
            self, drift_stream, length, output_path,
    ):
        command = self.generate_moa_command(drift_stream, length, output_path)
        subprocess.run(command, shell=True)

    #  @param streams: list of int, denoting order of streams in combination
    #  @param positions: list of int, center position of drift
    #  @param w_drift: int, width of drift
    #  @Returns drift_stream: string, MOA representation of drift stream
    #                     constructed according to given parameters
    def generate_drift_stream_for_moa(
            self, streams, positions, w_drift, streams_intermed
    ):
        """
        Create MOA representation of drift stream based on given parameters
        """
        N = len(positions)
        streams = streams[:N+1]
        w_drift = w_drift[:N]

        moa_streams = []
        # Get proper order for respective segments from ARFF files
        for (i, s) in enumerate(streams):
            moa_temp = self.get_stream_from_arff(streams_intermed[s][i])
            moa_streams.append(moa_temp)

        # Generate MOA stream using ARFF segments and parameters determined
        # from above
        drift_stream = self.generate_grad_stream_from_stream(
            moa_streams[0], moa_streams[1], positions[0], w_drift[0]
        )
        for i in range(1, len(moa_streams)):
            drift_stream = self.generate_grad_stream_from_stream(
                drift_stream, moa_streams[i], positions[i-1], w_drift[i-1]
            )
        return drift_stream

    #  @param w_drift: int, width of drift
    #  @param length: int, total length of new stream
    #  @param positions: list of int, center position of drift
    #  @param seq_before: list of boolean indicating whether drift comes before or before anomaly
    #  @param dataset: string, descriptor (name) or source dataset for identification
    #  @param sub_dir: string, name of subdirectory to export drift stream
    #  @Returns (output_path, filename): (string, string), output path for ARFF file and
    #                                general filename without file extension
    def get_output_filepath(
            self, w_drift, length, positions, seq_before, dataset, sub_dir
    ):
        """
        Evaluate true values for each parameter for file naming
        """
        p_drift = sum(w_drift)/length*100
        n_drift = len(positions)
        p_before = sum(seq_before)/len(seq_before)*100
        filename = f"{dataset}_grad_p{int(p_drift)}_n{n_drift}_b{int(p_before)}"
        output_path = f'{self.drift_dir}/{sub_dir}/{filename}.arff'
        return output_path, filename

    #  @param y: ndarray of shape (N,) corresponding to anomaly labels
    #  @Return list of lists denoting anomaly intervals in the form [start, end)
    def find_anomaly_intervals(self, y):
        """
        Method to find the intervals where there is an anomaly
        """
        change_indices = np.where(np.diff(y) != 0)[0]
        if len(change_indices) == 0:
            return []
        anom_intervals = []

        if y[change_indices[0]] == 0:
            i = 0
        else:
            i = 1
            anom_intervals.append([0, change_indices[0]+1])

        while i + 1 < len(change_indices):
            anom_intervals.append([change_indices[i]+1, change_indices[i+1]+1])
            i += 2

        if y[-1] == 1:
            anom_intervals.append([change_indices[-1]+1, len(y)])

        return anom_intervals

    #  @param anom_ints: list of list of int, anomaly intervals [start, end] of
    #                 input streams
    #  @Return total_anoms: list of tuple of (int, [int, int]),
    #      Represents (stream, [start, end]) for anomaly intervals across all
    #      streams ordered by start position
    def get_total_anoms(self, anom_ints):
        """
        Create list of anomaly intervals across all streams in increasing order
        of start position
        """
        total_anoms = []
        anom_ints_copy = [a_i.copy() for a_i in anom_ints]
        first_start = [anom_ints_copy[i][0] for i in range(len(anom_ints_copy))]
        while any(len(x) != 0 for x in anom_ints_copy):
            start_ints = [x[0] for x in first_start]
            i = np.argmin(start_ints)
            total_anoms.append((i, anom_ints_copy[i].pop(0)))
            if len(anom_ints_copy[i]) > 0:
                first_start[i] = anom_ints_copy[i][0]
            else:
                first_start[i] = [sys.maxsize, 0]
        return total_anoms

    #  @param filepath: String representing filepath of .arff file to split
    #  @param indices: list of int representing N-1 indices of original data to
    #                  split file
    #  @param split_name: String representing identifying name of split
    #  @param output_dir: String representing directory to write arff file
    #  @Return list of filepath
    def split_arff(self, filepath, indices, split_name, output_dir):
        """
        Write new .arff files which splits original file from index
        - File 0 contains points from index range [0, index_1)
        - File n contains points from index range [index_{n-1}, index_n) for 0 < n < N
        - File N contains points from index range [index_N - 1, file_length)
        """
        f = filepath.split('/')[-1]
        if output_dir is None:
            output_dir = filepath[:-len(f)]
        filename = f.split('.arff')[0]

        content = []
        output_files = []
        # Extract content from ARFF file
        with open(filepath, 'r') as input_file:
            for (i, line) in enumerate(input_file):
                if i > 7:
                    num_vals = line.strip().split(',')[:-1]
                    num_vals = [float(n) for n in num_vals]
                    newline = f"{num_vals[0]},{num_vals[1]},\n"
                    content.append(newline)
                else:
                    content.append(line)

        # Save the header information to write to each new split file
        header_lines = content[:7]
        for i in range(len(indices)):
            if i > 0:
                data = content[indices[i-1]+7:indices[i]+7]
            else:
                data = content[7:indices[i]+7]
            output_file_name = f"{output_dir}/{filename}_{split_name}_{i}.arff"
            with open(output_file_name, 'w') as output_file:
                output_file.writelines(header_lines + data)
            output_files.append(output_file_name)
        data = content[indices[-1]+7:]
        output_file_name = f"{output_dir}/{filename}_{split_name}_{len(indices)}.arff"

        # Write the data to a new file
        with open(output_file_name, 'w') as output_file:
            output_file.writelines(header_lines + data)
        output_files.append(output_file_name)
        return output_files

    #  @param moa_file_path: String, path to execute moa
    #  @param stream: String representing stream to be generated
    #  @Return string, command to run MOA through command line
    def generate_moa_command(self, stream, length, output_path):
        """
        Generate command to run with MOA CLI to create gradual stream
        """
        command_p1 = f'cd {self.moa_path} && java -cp moa.jar -javaagent:sizeofag-1.0.4.jar'
        command_p2 = \
            f'moa.DoTask "WriteStreamToARFFFile  -s ({stream}) -f {output_path} -m {length}"'
        return f'{command_p1} {command_p2}'

    #  @param stream_1: String, first stream in drift
    #  @param stream_2: String, second stream in drift
    #  @param position: int, center positions of drift
    #  @param width: int, width of drift
    #  @Return string, representation of resultant MOA stream
    def generate_grad_stream_from_stream(
            self,
            stream_1,
            stream_2,
            position,
            width
            ):
        """
        Generate command line representation of MOA ConceptDriftStream
        object (gradual drift)
        """
        drift_stream = \
            f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w {width}'
        return drift_stream

    #  @param file_path: String, filepath to source arff file
    #  @Return string, representation of MOA ArffFileStream
    def get_stream_from_arff(self, file_path):
        """
        Generate command line representation of MOA ArffFileStream object
        """
        return f'ArffFileStream -f {file_path} -c 0'

    #  @param stream_1: String, first stream in drift
    #  @param stream_2: String, second stream in drift
    #  @param position: int, center positions of drift
    #  @Return string, representation of resultant MOA stream
    def generate_abrupt_stream_from_stream(
            self,
            stream_1,
            stream_2,
            position
            ):
        """
        Generate command line representation of MOA ConceptDriftStream
        object (abrupt drift)
        """
        drift_stream = f'ConceptDriftStream -s ({stream_1}) -d ({stream_2}) -p {position} -w 1'
        return drift_stream

    #  @param positions: list of int, center position of drift
    #  @param w_drift: list of int, width of each drift
    #  @param length: int, total length of new stream
    #  @Return drift_label: list of int, whether or not a point is labelled as drift
    def get_drift_labels(self, positions, w_drift, length):
        """
        Returns drift labels of dataset, where 0 indicates no drift and 1
        indicates drift
        """
        drift_label = [0 for _ in range(length)]
        for (p, w) in list(zip(positions, w_drift)):
            drift_label[int(p - w/2):int(p + w/2)] = [1] * w
        return drift_label

    #  @param filepath: string, filepath to ARFF file to append info to
    #  @param source_files: list of string, filenames of source files
    #  @param streams: list of int, denotes the order of combination
    #  @param positions: list of int, center positions for all drifts
    #  @param seq_before: list of boolean, sequence showing relative
    #             position of drift with respect to anomaly
    def add_stream_info(
        self, filepath, source_files, streams, positions, w_drift, seq_before
    ):
        stream_info = ['%  Source Streams:']
        n = len(source_files)
        stream_info += [f'%    {i}:{source_files[i]}' for i in range(n)]
        stream_info.append(f'%  Stream Order: {streams}')
        stream_info.append(f'%  Drift Positions: {positions}')
        stream_info.append(f'%  Drift Widths: {w_drift}')
        stream_info.append(f'%  Drift Before: {seq_before}')
        with open(filepath, 'r+') as f:
            stream_data = f.read()
            f.seek(0)
            f.write('\n'.join(stream_info) + '\n' + stream_data)
