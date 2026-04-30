# Experiment Log

Use this file to record every meaningful run or data-processing decision.

## Template

### YYYY-MM-DD HH:MM - Run Name

- Purpose:
- Code version:
- Config:
- Dataset files:
- Train/validation/test split:
- Main parameters:
- Metrics:
- Figures/tables:
- Observations:
- Deviations from paper:
- Next action:

## Entries

### 2026-05-01 - Repository structure initialization

- Purpose: Create a reproducible project structure for NASA-only PI-TNet reproduction.
- Code version: Initial structured repository.
- Config: `configs/data/nasa.yaml`, `configs/model/pi_tnet.yaml`, `configs/experiment/pi_tnet_nasa_reproduction.yaml`.
- Next action: Audit raw NASA files and extract cycle-level discharge capacity.

### 2026-05-01 - M1 NASA data audit

- Purpose: Verify the NASA cells used by the PI-TNet paper before feature construction.
- Code version: `scripts/audit_nasa_data.py`, `src/battery_rul/data/nasa.py`.
- Config: `configs/data/nasa.yaml`.
- Dataset files: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` under `data/raw/1. BatteryAgingARC-FY08Q4`.
- Main parameters: nominal capacity = 2 Ah; EOL capacity = 1.4 Ah; cycle type = discharge.
- Metrics/artifacts: `data/processed/nasa_discharge_summary.csv`, `results/tables/nasa_cell_audit.csv`, `results/figures/nasa_capacity_degradation.png`, `results/logs/nasa_data_audit.json`.
- Observations: `B0005`, `B0006`, and `B0007` each contain 168 discharge cycles; `B0018` contains 132 discharge cycles in the raw NASA file.
- Deviations from paper: the paper table reports 168 cycles for `B0018`, but the audited raw NASA file contains 132 discharge records. Raw-data-derived counts are retained.
- Next action: construct fixed-length discharge-cycle feature sequences from voltage, current, temperature, and time.

### 2026-05-01 - M2 NASA feature construction

- Purpose: Build the PI-TNet input tensor from discharge-cycle voltage, current, temperature, and time.
- Code version: `scripts/build_nasa_features.py`, `src/battery_rul/data/nasa.py`.
- Config: `configs/data/nasa.yaml`.
- Dataset files: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`.
- Train/validation/test split: chronological 70/30 split per cell, matching the paper's first-70%-train and last-30%-test protocol. No validation split yet.
- Main parameters: 128 fixed sampling points per discharge cycle; features are voltage, current, temperature, and time; labels are capacity in Ah and SOH = capacity / 2 Ah.
- Metrics/artifacts: `data/processed/nasa_pi_tnet_features.npz`, `data/processed/nasa_pi_tnet_metadata.csv`, `data/splits/nasa_chronological_70_30_split.csv`, `results/logs/nasa_feature_build.json`.
- Observations: feature tensor shape is `(636, 4, 128)`. Split counts are `B0005`: 117 train / 51 test, `B0006`: 117 / 51, `B0007`: 117 / 51, `B0018`: 92 / 40.
- Deviations from paper: the paper does not disclose the exact fixed sampling length. NASA raw curves have variable sample lengths, so normalized-time interpolation to 128 points is used as a documented reproducibility assumption.
- Next action: implement the paper's baseline data loader and begin the PI-TNet model stack with CDP components.

### 2026-05-01 - M3 NASA training data loader

- Purpose: Implement paper-consistent PyTorch data loading for PI-TNet training and testing.
- Code version: `src/battery_rul/data/dataloaders.py`, `scripts/verify_nasa_dataloader.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Dataset files: `data/processed/nasa_pi_tnet_features.npz`, `data/processed/nasa_pi_tnet_metadata.csv`.
- Train/validation/test split: first 70% discharge cycles for training and last 30% for testing; no validation split introduced.
- Main parameters: batch size = 16; training loader shuffles; test loader preserves chronological order; feature tensor shape per sample is `(4, 128)`.
- Metrics/artifacts: `results/logs/nasa_dataloader_verification.json`.
- Observations: all-cell loader has 443 train / 193 test samples. Single-cell loaders match M2 split counts. First training batch shape is `(16, 4, 128)`.
- Deviations from paper: none for loader behavior. The feature interpolation assumption from M2 remains in effect.
- Next action: implement the PI-TNet Convolutional Data Processor components: CDC, HDC, VDC, and vanilla convolution.
