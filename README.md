# Adaptive Anomaly Detection in the Presence of Concept Drift
# Overview

This repository contains the implementation of **AnDri**, a time-series anomaly detection framework that jointly detects anomalies and adapts to evolving normal patterns through drift-aware normal pattern management.

Andri (Anomaly detection in the presence of Drift) is an adaptive, time-series, anomaly detection method cognizant of concept drift. AnDri co-detects anomalies and drift, extending the types of drift considered in anomaly detection to include gradual and recurring drifts.
- AnDri supports a dyanamic normal model where normal patterns are not fixed, but can be activated, deactivated or added over time. This adaptability enables AnDri to compute anomaly scores to the most similar active pattern.
- We introduce a new time-series clustering method, Adajcent Hierarchical Clustering (AHC), for learning normal patterns that respect their temporal locality; critical for detecting short-lived normal patterns that are overlooked by existing methods.

## Repository Structure

```text

.

├── data/                  # Processed datasets

├── results/               # Output directory

├── sample.ipynb           # Example notebook

├── test_andri.py          # Example script

├── util/

│   ├── ahc.py

│   ├── analy.py

│   ├── plot_andri.py

│   ├── util_andri.py

│   ├── util_data.py

│   ├── util_exp.py

│   └── TSB_AD/

│       ├── metrics.py

│       ├── slidingWindows.py

│       └── models/

│           ├── andri.py

│           ├── norma.py

│           ├── sand.py

│           └── error_detection.py

├── environment.yml

├── requirements.txt

└── README.md

```


## References of this repository
- https://github.com/TheDatumOrg/TSB-UAD
- https://github.com/imperial-qore/TranAD
- https://moa.cms.waikato.ac.nz
- https://github.com/mac-dsl/CanGene (To inject drifts)


## References
- TBA

## Contributors
- Jongjun Park

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

## Drift Injection
In this paper, we introduced drift injection using multiple ECG datasets. 

1. Gradual drift injection
![Image](https://github.com/user-attachments/assets/19b758fe-bdda-4630-8470-8283a2735862)
For drift injection, we used 4-different ECG data, ECG 803, 805, 806, and 820 from the above ECG datasets. Each has sligthly different ECG pattern, and anomalies in each data are also different from the others.

![Image](https://github.com/user-attachments/assets/85cc302f-5cb9-4933-b6bf-b0254f99c30e)
Using the MOA, our previous demo CanGene can inject gradual drifts acoording to near the anomaly points. The background colors represent the corresponding dataset (i.e., ECG 803 for blue, ECG 820 for purple, ...), yellow represents transition period for gradual drifts, and red indicates anomalies.

![Image](https://github.com/user-attachments/assets/b163a76e-753f-47a7-a5f9-4413a38db82c)
If we zoom-in one of the region, we can see the how the drift is simulated, i.e., taking one from the probability of sigmoid function (see the details in MOA or CanGene).

![Image](https://github.com/user-attachments/assets/81e25431-32bb-4e0c-8a57-56f9bc05a09b)
The heatmap shows the distances between normal patterns and corresponding selective anomalies. As we can see, some anomalies are quite similar to one of other normal patterns, leading false detection for some previous methods. 


2. Local anomaly injection
![Image](https://github.com/user-attachments/assets/3e0f3b36-4837-4349-8144-a3b80d49c319)
After removing anomalies in two ECG datasets and aligning the time-series, we concatenate them and put one subsequence into the other time interval, to simulating local-anomalies. (Gloally, they looks like normal pattern, but locally rare enough).

![Image](https://github.com/user-attachments/assets/e3fd274c-24c0-44d3-b37c-68685f1de89e)



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

  
