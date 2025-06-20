# AnDri: Anomaly detection in the presence of Drift.
# Overview
![Image](https://github.com/user-attachments/assets/a330fad0-3023-40bd-b4ac-eeefcfab6b1b)

Andri (Anomaly detection in the presence of Drift) is an adaptive, time-series, anomaly detection method cognizant of concept drift. AnDri co-detects anomalies and drift, extending the types of drift considered in anomaly detection to include gradual and recurring drifts.
- AnDri supports a dyanamic normal model where normal patterns are not fixed, but can be activated, deactivated or added over time. This adaptability enables AnDri to compute anomaly scores to the most similar active pattern.
- We introduce a new time-series clustering method, Adajcent Hierarchical Clustering (AHC), for learning normal patterns that respect their temporal locality; critical for detecting short-lived normal patterns that are overlooked by existing methods.

# Example of AHC
![Image](https://github.com/user-attachments/assets/4b7de77d-fbfd-4d74-bd70-5e5b1c103beb)
For this example, there are two normal patterns (ECG 803, ECG 805) are repeatedly appeared, with different frequency. First, we divided it into the length of normal pattern (i.e., 2xl), and compare similarities between them for only adjacent subsequences (i.e., left and right)

![Image](https://github.com/user-attachments/assets/1dea8303-9686-4d99-afc4-f11942c2a6cc)
After that, AHC merges sub-clusters using WARD-linkage distance, in an aggremerative hierarchical manner. 

![Image](https://github.com/user-attachments/assets/409deb2c-2fcb-4eec-baf8-1083323fa2eb)
When a new linkage distance is less than prev. linkage distance, we called it reversion, AHC rolls back the previous merge, and revise the merge, even if those are not adjacent each other. 


## References of this repository
- https://github.com/TheDatumOrg/TSB-UAD
- https://github.com/imperial-qore/TranAD
- https://moa.cms.waikato.ac.nz
- https://github.com/mac-dsl/CanGene (To inject drifts)


## References
- Will be apeared

## Contributors
- Will be apeared

## Installation

Steps:

1. Clone the repository git

```
git clone https://github.com/mac-dsl/AnDri.git
```

2. Install dependencies from requirement.txt

```
pip install -r requirements.txt
```

## Datasets
All are datasets and time series are stored in ./data. We describe below the different types of datasets used in our benchmark.
1. ECG
- The ECG dataset is a standard electrocardiogram dataset which comes from the MIT-BIH Arrhythmia database, where anomalies represent ventricular premature contractions. (Downloaded from https://github.com/TheDatumOrg/TSB-UAD/tree/main)

2. IOPS
- The IOPS dataset is a set of performance indicators reflecting the scale and performance of web services. (Downloaded from https://github.com/TheDatumOrg/TSB-UAD/tree/main)

3. Elec
- The Elec. dataset is a half-hourly aggregated electricity usage patterns in New South Wales, Australia.  (Downloaded from https://moa.cms.waikato.ac.nz)
  
4. Weather data
- The Weather dataset is a hourly, geographically aggregated temperature and radiation information in Europe originated from the NASA MERRA-2. The original Weather dataset contained timestamp, each country's temperature, and radiation information from 1960-to-2020, but for the demo, we separated each country's temperature data from 2017-to-2020 and saved them into '.arff' files in the repository. (From Open Power System Data, https://doi.org/10.25832/weather_data/2020-09-16:)


## Usage
We include main algorithms, 
- Adjacent Hierarchical Clustering (/util/ahc.py)
- AnDri (/util/TSB_AD/models/andri.py)
along with a simple example to show how it runs. (sample.ipynb)

sample.ipynb
- This notebook includes step-by-step procedure of running AnDri.
- Using anomaly injected Elec. dataset (one-sample with 10% anomaly injection over uniform distribution), we computed AnDri (offline) and (online)

AnDri's parameters:
- 'normalize': we support three distance metrics, (1) 'z-norm': Z-normalized distance, (2) 'zero-mean': Z-norm distance without devision by standard deviation, (3) 'Euclidean'
- 'kadj': k-AHC, k-hop distance to compare
- 'nm_len': length of normal pattern (nm_len x l (slidingWindow))
- 'min_size': same paramter of R_{min}, minimum size of cluster
- 'max_W': maximum length of moving window for detecting concept drift
- 'train_len': Length of trianing set (i.e., 0.2 for 20%)
- 

  
