import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# injects a random sequence of anomalies (i.e collective anomaly)
class CollectiveAnomaly:
    def __init__(self, length: int, percentage: float, distribution: str = 'uniform', upperbound: float = None, lowerbound: float = None, mu: float = None, sigma: float = None, skew=None, num_values: int = 5) -> None:
        # if upperbound and lowerbound not given, then random sequence anomaly can be generated using random values already present in the data
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.length = length  # specifies the length of the random sequence
        self.percentage = percentage
        self.dist = distribution
        self.mean = mu
        self.std = sigma
        self.num_values = num_values
        self.skew = skew

# injects point anomalies; these deviate significantly from the rest of the data
class PointAnomaly:
    def __init__(self, percentage: float, distribution: str = 'uniform', mu: float = None, sigma: float = None, num_values: int = 5, lowerbound=None, upperbound=None, skew=None) -> None:
        self.percentage = percentage
        self.dist = distribution
        self.mean = mu
        self.std = sigma
        self.num_values = num_values
        self.skew = skew
        self.lowerbound = lowerbound
        self.upperbound = upperbound

# injects sequential anomalies into the dataset (collective anomalies that keep repeating)
class PeriodicAnomaly:
    def __init__(self, percentage: float, noise_factor: float, start=None, end=None, length=15):
        self.length = length
        self.percentage = percentage
        self.noise_factor = noise_factor
        self.start = start
        self.end = end
