import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import skewnorm
from util.anomaly import CollectiveAnomaly, PointAnomaly, PeriodicAnomaly
from util.stream import Stream


class createAnomalyIntervals:
    def __init__(self, stream: Stream) -> None:
        self.stream = stream
        self.dataset = stream.data
        self.stream_anomaly_labels = stream.anomaly_labels
        self.num_intervals = None
        self.points = []
        # debugging purposes, remove later
        # print(type(self.dataset), self.dataset.size)
        # print(type(self.stream_anomaly_labels), self.stream_anomaly_labels.size)

    def create_intervals(self, num_intervals: int, gap_size: int):
        starting_points = []  # contains the starting point for each drift interval
        ending_points = []  # contains the ending point for each drift interval
        self.num_intervals = num_intervals
        evenly_spaced = np.linspace(0, len(self.dataset), num=(num_intervals+1))
        #print(evenly_spaced)# debugging
        starting_points.append(0+1/2*gap_size)
        points = []
        for i in range(1, len(evenly_spaced)):
            ending_points.append(evenly_spaced[i]-(gap_size/2))
        for i in range(1, len(evenly_spaced)-1):
            starting_points.append(evenly_spaced[i]+(gap_size/2))
        for i in range(0, len(starting_points)):
            points.append((starting_points[i], ending_points[i]))
        self.points = points
        
        

    def add_anomalies(self, *anomaly_modules):
        if len(anomaly_modules) != len(self.points):
            # throw exception here because the number of anomaly modules must be the same as the number of intervals
            raise ValueError(
                'The number of anomaly modules given is not the same as the number of intervals specified.')
        for i in range(0, len(self.points)):
            # depending on type of anomaly module, the insertion parameters are different
            if type(anomaly_modules[i]) == PointAnomaly:
                self.add_dist_point_anomaly(
                    self.points[i][0], self.points[i][1], anomaly_modules[i].percentage, anomaly_modules[i].dist,
                    anomaly_modules[i].mean, anomaly_modules[i].std, anomaly_modules[i].num_values,
                    anomaly_modules[i].upperbound, anomaly_modules[i].lowerbound, anomaly_modules[i].skew)

            elif type(anomaly_modules[i]) == CollectiveAnomaly:
                self.add_Collective_Anomaly(
                    self.points[i][0], self.points[i][1], anomaly_modules[i].length, anomaly_modules[i].percentage,
                    anomaly_modules[i].dist, anomaly_modules[i].mean, anomaly_modules[i].std, anomaly_modules[i].num_values,
                    anomaly_modules[i].upperbound, anomaly_modules[i].lowerbound, anomaly_modules[i].skew
                )
            elif type(anomaly_modules[i]) == PeriodicAnomaly:
                self.add_sequential_anomaly(
                    self.points[i][0], self.points[i][1], anomaly_modules[i].percentage,
                    anomaly_modules[i].noise_factor, anomaly_modules[i].start, anomaly_modules[i].end, anomaly_modules[i].length
                )
            else:
                raise ValueError(
                    "Wrong type of input parameter, must be anomaly modules.")
        #self.stream.__set_anomaly_intervals()

    # adds point anomalies within specified intervals
    def add_Point_Anomaly(self, start: int, end: int, percentage: float, possible_values: list = None) -> None:
        insertion_indexes = np.random.choice(
            np.arange(start, end), int(percentage*(end-start)))
        for index in insertion_indexes:
            self.dataset[index] = self.dataset[index] * np.random.choice(possible_values)  # setting the anomaly
            self.stream_anomaly_labels[index] = 1  # setting the label as anomalous
        self.stream.data = self.dataset
        self.stream.anomaly_labels = self.stream_anomaly_labels

    # adds point anomalies according to a distribution
    def add_dist_point_anomaly(self, start: int, end: int, percentage: float, distribution, mu, std, num_values, upperbound, lowerbound, skew):
        if mu == None:
            mu = self.dataset[int(start):int(end)].mean() * 3
        if std == None:
            std = self.dataset[int(start):int(end)].std() * 3
        if distribution == 'uniform':
            possible_values = np.random.uniform(
                lowerbound, upperbound, num_values)
        elif distribution == 'skew':
            possible_values = skewnorm.rvs(
                a=skew, loc=upperbound, size=num_values)
            possible_values = possible_values - min(possible_values)
            possible_values = possible_values / max(possible_values)
            possible_values = possible_values * upperbound

        elif distribution == 'gaussian':
            possible_values = np.random.normal(mu, std, num_values)
        else:
            raise ValueError(
                'Wrong distribution specification. Please enter either uniform, skew, or gaussian in string format')

        insertion_indexes = np.random.choice(
            np.arange(start, end-1), int(percentage*(end-start)))
        #print("Insertion Indexes:" + str(insertion_indexes))

        for index in insertion_indexes:
            self.dataset[int(index)] += self.dataset[int(index)] * np.random.choice(possible_values)
            self.stream_anomaly_labels[int(index)] = 1
        
        self.stream.data = self.dataset
        self.stream.anomaly_labels = self.stream_anomaly_labels

    def add_Collective_Anomaly(self, start: int, end: int, length: int, percentage: float, distribution, mu, std, num_values, upperbound, lowerbound, skew):

        number_anomalies = math.ceil(((end-start)/length)*percentage)

        if mu == None:
            mu = self.dataset[int(start):int(end)].mean() * 3
        if std == None:
            std = self.dataset[int(start):int(end)].std() * 3
        if distribution == 'uniform':
            possible_values = np.random.uniform(
                lowerbound, upperbound, num_values)
        elif distribution == 'skew':
            possible_values = skewnorm.rvs(
                a=skew, loc=upperbound, size=num_values)
            possible_values = possible_values - min(possible_values)
            possible_values = possible_values / max(possible_values)
            possible_values = possible_values * upperbound

        elif distribution == 'gaussian':
            possible_values = np.random.normal(mu, std, num_values)
        else:
            raise ValueError(
                'Wrong distribution specification. Please enter either uniform, skew, or gaussian')

        insertion_indexes = np.random.choice(
            np.arange(start, end, length), number_anomalies)
        # creating the collective sequence
        collective_sequences = []
        for _ in range(number_anomalies):
            collective_sequences.append(
                np.random.choice(possible_values, length))
        
        # debugging 
        # print(insertion_indexes)
        # print("COLLECTIVE SEQUENCES")
        #print(collective_sequences[0].reshape(-1, 1).shape)
        # print(self.dataset[int(insertion_indexes[0]): int(insertion_indexes[0]) + length].shape)
       

        # inserting collective anomalies at required index
        for i in range(0, len(insertion_indexes)):
            reshaped_collective = collective_sequences[i].reshape(-1, 1)
            

            self.dataset[int(insertion_indexes[i]): int(insertion_indexes[i]) + length] += \
            np.multiply(reshaped_collective, self.dataset[int(insertion_indexes[i]): int(
                    insertion_indexes[i]) + length]) 
            
            # setting the label as anomalous
            self.stream_anomaly_labels[int(insertion_indexes[i]): int(
                insertion_indexes[i]) + length] = 1
            
        self.stream.data = self.dataset
        self.stream.anomaly_labels = self.stream_anomaly_labels


    def add_sequential_anomaly(self, start, end, percentage, noise_factor, starting, ending, length):
        if starting == None and ending == None:
            starting = int(np.random.choice(np.arange(start, end-length)))
            anomaly_sequence = self.dataset[starting:starting + length]

        if ending == None:
            anomaly_sequence = self.dataset[starting:starting + length]
        else:
            anomaly_sequence = self.dataset[starting:ending]
            length = ending - starting

        num_anomalies = math.ceil(((end-start)/length)*percentage)

        # mid_insertions = np.linspace(start, end, num_anomalies)
        # insertion_indexes = []
        # for index in mid_insertions:
            # insertion_indexes.append(int(math.ceil(index-length/2)))
        insertion_indexes = np.random.choice(np.arange(start, end, length), num_anomalies)

        # add noise processing here
        noise = np.random.normal(0, noise_factor, len(anomaly_sequence))
        # print(noise)
        # anomaly_sequence = anomaly_sequence + noise
        # revise anomaly sequence which looks similar to the original seq. [start:end] # jp 20240531
        anomaly_sequence = anomaly_sequence.reshape(-1) + noise.reshape(-1)
        anomaly_sequence = anomaly_sequence.reshape(-1,1)

        # print("DATASET SLICE")
        # print(self.dataset[int(insertion_indexes[0]): int(insertion_indexes[0]) + length].shape)
        # print(anomaly_sequence[0])
        # print(anomaly_sequence[1])

        # insertine sequential anomalies at required index
        for i in range(0, len(insertion_indexes)):
            self.dataset[int(insertion_indexes[i]): int(
                # insertion_indexes[i]) + length] = anomaly_sequence[0].reshape(-1, 1)
                # revise anomaly sequence which looks similar to the original seq. [start:end] # jp 20240531
                insertion_indexes[i]) + length] = anomaly_sequence.reshape(-1, 1) 
            # setting the label as anomalous
            self.stream_anomaly_labels[int(insertion_indexes[i]): int(
                insertion_indexes[i]) + length] = 1
        self.stream.data = self.dataset
        self.stream.anomaly_labels = self.stream_anomaly_labels

    def plot_dataset(self):
        plt.figure(figsize=(100, 30))
        plt.plot(self.dataset)
        for point in self.points:
            plt.axvline(x=point[0], color='r', linestyle="--", linewidth=4)
            plt.axvline(x=point[1], color='r', linestyle="--", linewidth=4)
        plt.show()
