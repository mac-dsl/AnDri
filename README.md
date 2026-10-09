# Adaptive Anomaly Detection in the Presence of Concept Drift

AnDri detects time series anomalies while adapting its normal patterns to concept drift. Its adaptive adjacent hierarchical clustering (AHC) supports gradual and recurring changes.

<img width="2918" height="1133" alt="AnDri overview" src="https://github.com/user-attachments/assets/57b484df-baa0-4547-bcb1-b7daae22209d" />

## Repository structure

```text
.
├── data/processed/        # Ready-to-run dataset CSVs
│   ├── climate/
│   ├── traffic/
│   ├── NAB/
│   ├── Environ/
│   ├── CATSv2/
│   └── SMD/
├── util/
│   ├── ahc.py             # Adaptive AHC and timing breakdown
│   ├── util_exp.py        # Dataset loading and evaluation
│   └── TSB_AD/models/andri.py
├── test_andri.py          # Offline/online experiment entry point
├── sample.ipynb           # Step-by-step example
├── draw_figure.ipynb      # Paper figure notebook
├── results/tested/        # Committed figure/table inputs
├── png/                   # Generated paper figures (PDF)
├── requirements.txt
└── environment.yml
```

`data/processed/` contains the current experiment inputs. `test_andri.py` reads `climate/*_processed.csv` (`data`, `label`), `traffic/*.csv` (`total_flow`, `total_flow_label`), and NAB/Environ CSVs (`Data`, `Label`). CATSv2 CSVs contain a `label` column and sensor columns, which are evaluated separately. The dataset sources are listed in [data/processed/README_Data.md](data/processed/README_Data.md).

## Installation

```bash
conda env create -f environment.yml
conda activate andri_repo
```

Alternatively, use Python 3.11 and `pip install -r requirements.txt`.

## Usage

Run the script from the repository root. One invocation processes **all matching input CSV files** in the selected dataset folder. For CATSv2, it runs each sensor column of every CSV separately.

### Data setup

Place the input files under `data/processed/`. The script reads these columns and file patterns:

```text
data/processed/climate/*_processed.csv   data, label
data/processed/traffic/*.csv             total_flow, total_flow_label
data/processed/NAB/*.csv                 Data, Label
data/processed/Environ/*.csv             Data, Label
data/processed/CATSv2/*.csv              label + one or more sensor columns
```

Choose one dataset per command:

```bash
python test_andri.py -data climate
python test_andri.py -data traffic
python test_andri.py -data NAB
python test_andri.py -data Environ
python test_andri.py -data CATSv2
```

Climate and traffic use a 24-point window and the first 8,760 points for training. NAB, Environ, and CATSv2 take the training boundary from `_tr_<number>` in each filename and choose the window length from the data.

### Parameter setup

`adaptive_ahc` is the default and runs both offline and online modes. `hc` (independent hierarchical clustering) and `kshape` run offline only. Select a clustering method and its parameters in the command:

```bash
python test_andri.py -data climate -clustering adaptive_ahc -k 5 -nm_len 2
python test_andri.py -data NAB -clustering hc -k 5 -linkage ward
python test_andri.py -data traffic -clustering kshape -k 5
```

`-k` is the AHC neighborhood size for `adaptive_ahc`, or the cluster count for `hc` and `kshape`. Other options are `-normalize`, `-max_W`, `-delta_max`, `-rmin`, `-step`, and `-rollback`. See their defaults with:

```bash
python test_andri.py -h
```

### Result data

Each run writes its files under `results/<dataset>/`. The input name, normal-pattern length, cluster setting, and optional clustering method are included in the filename. For CATSv2, the sensor name is included as well.

```text
results/<dataset>/
  AnDri_<input>_nm_<n>_k_<k>_clf_off.pickle       # offline model
  AnDri_<input>_nm_<n>_k_<k>_clf_on.pickle        # online model, AHC only
  AnDri_<input>_nm_<n>_k_<k>_scores_rev.pickle    # normalized score arrays
  AnDri_<input>_nm_<n>_k_<k>_results_org.csv      # evaluation metrics
  time_all.csv                                    # AHC wall-clock seconds and flips
  time_all_hc.csv / time_all_kshape.csv           # offline wall-clock seconds
```

For `hc` and `kshape`, filenames also contain the clustering method. Each timing CSV has one row per processed input (one row per sensor for CATSv2). The saved model's `runtime` attribute contains detailed stage timings; adaptive AHC also records stage timings in `util.ahc.elap_times`. Generated experiment results are excluded from Git; `results/tested/` contains the committed paper figure and table inputs.

## Paper figures

`draw_figure.ipynb` reads the saved tables and plot inputs in `results/tested/` and draws the paper figures. The figure PDFs are in `png/`. The notebook uses the saved CSVs and does not rerun model fitting.

## References of Repository

[TSB-UAD](https://github.com/TheDatumOrg/TSB-UAD), [TranAD](https://github.com/imperial-qore/TranAD), [ARCUS](https://github.com/kaist-dmlab/ARCUS), [DIVAD](https://github.com/exathlonbenchmark/divad), [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly), [METER](https://github.com/zjiaqi725/METER), and [CANDI](https://github.com/kimanki/CANDI).
