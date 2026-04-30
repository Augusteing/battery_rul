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
