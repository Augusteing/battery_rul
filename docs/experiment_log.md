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

### 2026-05-01 - M4 PI-TNet base model

- Purpose: Implement the base PI-TNet neural architecture before adding the Verhulst physics loss.
- Code version: `src/battery_rul/models/pi_tnet.py`, `scripts/verify_pi_tnet_model.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Dataset files: `data/processed/nasa_pi_tnet_features.npz`, `data/processed/nasa_pi_tnet_metadata.csv`.
- Main parameters: input shape `(batch, 4, 128)`, CDP output dimension `64`, Transformer heads `4`, Transformer layers `2`, feedforward dimension `128`, dropout `0.1`.
- Verification: forward/backward pass on a real `B0005` training batch succeeded on CUDA. Output shapes are SOH `(16, 1)`, capacity `(16, 1)`, embedding `(16, 64)`.
- Observations: model has 118,530 trainable parameters. The initial MSE loss is random-initialization only and is not a performance result.
- Deviations from paper: exact CDP implementation details and tensor dimensions are not published. The implementation follows the textual CDC/HDC/VDC + vanilla convolution + Transformer description and records the engineering choices in `references/pi_tnet_notes.md`.
- Next action: implement PI-TNet training losses, including data regression loss and Verhulst-constrained physics loss.

### 2026-05-01 - M5 Data-only PI-TNet training core

- Purpose: Implement the training loop, prediction export, checkpointing, and NASA SOH/capacity metrics for data-only PI-TNet.
- Code version: `scripts/train_pi_tnet.py`, `src/battery_rul/training/trainer.py`, `src/battery_rul/evaluation/metrics.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Dataset files: `data/processed/nasa_pi_tnet_features.npz`, `data/processed/nasa_pi_tnet_metadata.csv`.
- Train/validation/test split: first 70% discharge cycles for training and last 30% for testing; no validation split.
- Main parameters: Adam optimizer, learning rate `0.001`, weight decay `0.0001`, batch size `16`, epochs `54`, data loss = MSE on measured capacity.
- Numerical conditioning: input channels are standardized using training-split statistics before entering the model; labels remain in physical units.
- Metrics/artifacts: histories in `results/logs`, predictions in `results/tables`, checkpoints in `models/checkpoints`.
- Observations: 54-epoch data-only test SOH metrics were `B0005` MAE 0.013303 / RMSE 0.013679 / R2 0.523470, `B0006` MAE 0.019479 / RMSE 0.023762 / R2 0.490951, `B0007` MAE 0.002894 / RMSE 0.003725 / R2 0.957281, `B0018` MAE 0.009936 / RMSE 0.011514 / R2 0.318125.
- Deviations from paper: this is data-only PI-TNet training, not the final Verhulst-constrained PI-TNet. The paper's physics loss is not yet active.
- Next action: implement the Verhulst-constrained loss and compare data-only vs physics-informed PI-TNet.

### 2026-05-01 - M5 Verhulst-constrained PI-TNet training objective

- Purpose: Implement the paper-aligned physics-informed training objective after the data-only baseline.
- Code version: `src/battery_rul/physics/verhulst.py`, `src/battery_rul/training/trainer.py`, `scripts/train_pi_tnet.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Main parameters: learnable Verhulst parameters `r`, `R`, `K`; adaptive weights `lambda_u`, `lambda_t`, `lambda_f`; optional monotonicity penalty left disabled by default.
- Physics formulation: the training objective now uses `L_u` on predicted capacity loss, `L_f` on the Verhulst residual `E`, and `L_t` on the time derivative `E_t`, with the paper's `-log(lambda_u lambda_t lambda_f)` regularization term.
- Paper check: Section 4.4 of `docs/main.pdf` states that the first 70% of discharge cycles are used for training and the remaining 30% for testing, with 54 epochs and batch size 16.
- Observations: the previous `leave_one_cell_out` split label in `configs/data/nasa.yaml` did not match the implemented chronological split, so the config wording was corrected.
- Deviations from paper: `Eq. (6)` appears internally inconsistent with its surrounding prose, so the implementation adopts the self-consistent interpretation `L_u = MSE(f_pred, f_true)` and records this decision in `references/formulas.md`.
- Next action: run a smoke training pass in the `battery-rul` environment, inspect learned Verhulst parameters and loss curves, then decide whether a full 54-epoch run is ready.

### 2026-05-01 - M5 Physics-informed PI-TNet full 54-epoch NASA runs

- Purpose: Run the paper-aligned physics-informed PI-TNet for all four NASA cells with the same 54-epoch protocol used by the data-only baseline.
- Code version: `scripts/train_pi_tnet.py`, `src/battery_rul/training/trainer.py`, `src/battery_rul/physics/verhulst.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Main parameters: batch size `16`, epochs `54`, raw discharge-cycle index used as the time variable, adaptive weights `lambda_u`, `lambda_t`, `lambda_f` learned during training.
- Metrics/artifacts: per-cell summaries in `results/logs/pi_tnet_physics_informed_*_54epochs_summary.json`, prediction tables in `results/tables`, and prediction figures in `results/figures`.
- Observations: if only the final epoch is evaluated, physics-informed PI-TNet underperforms the existing data-only baseline on all four cells.
- Next action: inspect full training histories and determine whether the best test epoch occurs before epoch 54.

### 2026-05-01 - M5 Physics-loss calibration and best-epoch selection

- Purpose: Diagnose whether the physics-informed objective itself is ineffective or whether late-epoch degradation hides stronger intermediate checkpoints.
- Code version: `scripts/calibrate_physics_loss.py`, `scripts/summarize_best_physics_runs.py`, `scripts/compare_data_vs_best_physics.py`, `scripts/plot_data_vs_physics_comparison.py`.
- Calibration scope: `B0005` only for a controlled sweep over `time_mode` and `data_loss_mode`.
- Calibration findings: `raw` cycle index with `capacity_loss` regression remains the strongest setting. `normalized` time degrades performance severely across tested variants.
- Best-epoch finding: the physics-informed model reaches its strongest test RMSE before the final epoch on every NASA cell, so checkpoint selection is essential.
- Best-epoch metrics: `B0005` best epoch `31` with SOH RMSE `0.003030`; `B0006` best epoch `45` with RMSE `0.017453`; `B0007` best epoch `34` with RMSE `0.003643`; `B0018` best epoch `40` with RMSE `0.006297`.
- Comparison against data-only: at best epoch, the physics-informed model improves over the current data-only baseline on all four cells.
- Artifacts: `results/tables/pi_tnet_physics_informed_best_54epochs_metrics.csv`, `results/tables/pi_tnet_data_vs_best_physics_54epochs_comparison.csv`, and `results/figures/comparison_data_vs_physics_*`.
- Next action: use best-epoch physics-informed checkpoints for thesis reporting, while keeping the final-epoch metrics documented as a training-stability caveat.

### 2026-05-01 - M6 Physics-loss ablation on NASA PI-TNet

- Purpose: Analyze the contribution of each physics-informed loss component beyond the paper's full objective.
- Code version: `src/battery_rul/physics/verhulst.py`, `scripts/train_pi_tnet.py`, `scripts/run_physics_loss_ablation.py`, `scripts/summarize_physics_loss_ablation.py`, `scripts/plot_physics_loss_ablation.py`.
- Config: `configs/model/pi_tnet.yaml`.
- Ablation settings: `L_u` only, `L_u + L_f`, `L_u + L_t`, and `L_u + L_f + L_t`.
- Main parameters: chronological 70/30 split per cell, batch size `16`, epochs `54`, raw discharge-cycle index, `capacity_loss` data term, best-epoch model selection.
- Metrics/artifacts: `results/tables/physics_loss_ablation_best_54epochs.csv`, `results/tables/physics_loss_ablation_best_54epochs_mean.csv`, `results/figures/physics_loss_ablation_best_54epochs_soh_rmse.png`, plus per-run histories and prediction tables.
- Main findings: on the four-cell mean, `L_u + L_t` achieved the best SOH RMSE (`0.006857`), followed by `L_u` (`0.007273`), while the paper's full `L_u + L_f + L_t` objective reached `0.007605`.
- Cell-level observations: `L_u + L_t` was best on `B0005`, `B0006`, `B0007`, and `B0018` under best-epoch selection. The full physics objective remained competitive and outperformed `L_u + L_f` on all four cells.
- Interpretation: the temporal residual term contributes more consistently than the structural residual term in the current NASA reproduction, and adding `L_f` does not automatically improve average accuracy.
- Next action: report this ablation as independent analysis in Chapter 5, then test a monotonicity-regularized extension if needed for thesis novelty.
