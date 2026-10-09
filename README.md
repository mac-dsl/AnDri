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

Run from the repository root:

```bash
python test_andri.py -data climate
python test_andri.py -data NAB -clustering hc -k 5
python test_andri.py -data traffic -clustering kshape -k 5
```

`-data` accepts `climate`, `traffic`, `NAB`, `Environ`, or `CATSv2`. The current script selects the first sorted input file for each dataset; for CATSv2 it selects sensor data from `138_CATSv2_id_1_Sensor_tr_16568_1st_16668_subset.csv`. It does not run a whole-dataset benchmark in one invocation. Climate and traffic use a 24-point window and the first 8,760 points for training. Other datasets use the `_tr_` boundary in the filename and an automatically selected window length.

`-clustering` defaults to `adaptive_ahc`, which runs both offline and online AnDri. `hc` uses the independent hierarchical clustering implementation, and `kshape` uses KShape; both run offline only. Other options include `-nm_len` (normal pattern length multiplier), `-normalize`, `-k` (cluster count for `hc`/`kshape`, neighborhood size for AHC), `-linkage`, `-max_W`, `-delta_max`, `-rmin`, `-step`, and `-rollback`. Run `python test_andri.py -h` for their defaults.

Results are written under `results/<dataset>/`. `time_all.csv` (or `time_all_hc.csv` / `time_all_kshape.csv`) records elapsed wall-clock seconds for each run and, for AHC, the offline and online flip counts. The model also stores stage timing lists in `model.runtime`; AHC records its initialization, rollback, normal-model computation, and total times in `util.ahc.elap_times`. These are detailed measurements, separate from the script's wall-clock totals. Generated experiment results are local and excluded from Git.

## Paper figures

`draw_figure.ipynb` reads the saved tables and plot inputs in `results/tested/` and draws the paper figures. The figure PDFs are in `png/`. The notebook uses the saved CSVs and does not rerun model fitting.

## Related projects

[TSB-UAD](https://github.com/TheDatumOrg/TSB-UAD), [TranAD](https://github.com/imperial-qore/TranAD), [ARCUS](https://github.com/kaist-dmlab/ARCUS), [DIVAD](https://github.com/exathlonbenchmark/divad), and [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly).
