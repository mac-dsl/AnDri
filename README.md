# AnDri: Anomaly detection in the presence of Drift.

## References of this repository
- https://github.com/TheDatumOrg/TSB-UAD
- https://github.com/imperial-qore/TranAD
- https://moa.cms.waikato.ac.nz


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

## Benchmark
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

  
