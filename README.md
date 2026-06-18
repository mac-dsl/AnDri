# Adaptive Anomaly Detection in the Presence of Concept Drift
# Overview

This repository contains the implementation of **AnDri**, a time-series anomaly detection framework that jointly detects anomalies and adapts to evolving normal patterns through drift-aware normal pattern management.

<img width="2918" height="1133" alt="overview" src="https://github.com/user-attachments/assets/57b484df-baa0-4547-bcb1-b7daae22209d" />

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
│           └── andri.py
├── environment.yml
├── requirements.txt
└── README.md
```

## References of this repository
- https://github.com/TheDatumOrg/TSB-UAD
- https://github.com/imperial-qore/TranAD
- https://moa.cms.waikato.ac.nz
- https://github.com/szamani20/time-series
- https://github.com/kaist-dmlab/ARCUS
- https://github.com/exathlonbenchmark/divad
- https://github.com/NetManAIOps/OmniAnomaly


## References
- TBA

## Contributors
- Jongjun Park

## Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate andri_repo
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## Quick Start

Run test_andri.py

```bash
python3 test_andri.py -data climate -method AnDri
```

## Data

Example datasets are organized under:

```
data/
└── processed/
    ├── 2021_2025_precip_selected/
    ├── PeMS/
    ├── real iot/
    └── SWaT/
```

## Results

Generated anomaly scores, figures, and other outputs are saved in the `results/` directory.

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

  
