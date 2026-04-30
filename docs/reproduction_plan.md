# PI-TNet NASA Reproduction Plan

## Paper Target

Primary paper: *Integrating transformers into physics-informed neural networks: An approach to lithium-ion battery state-of-health prognostics*.

The first reproduction stage focuses only on the NASA battery aging dataset. Any extension to CALCE or other datasets should be treated as a later experiment.

## NASA Protocol Extracted from the Paper

- Cells: `B0005`, `B0006`, `B0007`, `B0018`.
- Inputs: discharge-cycle voltage, current, temperature, and timestamp.
- Target: measured capacity / SOH.
- Rated capacity: 2 Ah.
- Ambient temperature: 24 C.
- Discharge current: 2 A.
- End voltages: 2.7 V for `B0005`, 2.5 V for `B0006` and `B0018`, 2.2 V for `B0007`.
- Paper table reports 168 cycles for all four cells.

Data-audit note: the downloaded NASA `B0018.mat` file contains 132 discharge cycles, while `B0005`, `B0006`, and `B0007` contain 168 discharge cycles. This discrepancy is recorded and raw-data-derived values are used unless a later paper-specific correction is justified.

## Reproduction Principles

1. Raw data are immutable.
2. Every derived table must be produced by a script.
3. Every experiment must have a YAML configuration snapshot.
4. Metrics and figures must be saved in machine-readable and thesis-ready formats.
5. Deviations from the paper must be recorded explicitly.

## Planned Milestones

### M1: Data Audit

- Locate NASA `.mat` files.
- Confirm available cells and cycle counts.
- Extract discharge capacity per cycle.
- Plot capacity and SOH degradation curves.

### M2: Feature Construction

- Convert variable-length discharge curves into fixed-length sequences.
- Build feature tensors from voltage, current, temperature, and time.
- Define SOH as measured capacity divided by nominal capacity.
- Follow the paper's chronological split: the first 70% discharge cycles are used for training and the remaining 30% for testing; the training set is shuffled during model training.
- Reproducibility assumption: because the paper states that measurements are compared at identical sampling points but does not disclose the exact number of points, each discharge curve is interpolated to a fixed normalized-time grid. The current configuration uses 128 points per cycle and keeps raw physical units.

### M3: Baseline Models

- Implement paper-consistent PyTorch data loaders.
- Use the first 70% discharge cycles for training and the remaining 30% for testing.
- Shuffle only the training loader.
- Use batch size 16 and 54 epochs as reported in the implementation details.
- Establish reproducible metrics for the paper's per-cell NASA evaluation.

### M4: Physics-Informed Model

- Implement the physics residual loss using the selected degradation model.
- Add monotonic degradation regularization if required.
- Compare data-only and physics-informed training.

### M5: PI-TNet

- Implement Transformer encoder architecture.
- Integrate data loss and physics loss.
- Reproduce main curves and metric tables.

### M6: Thesis Outputs

- Export all final figures.
- Generate metric tables.
- Summarize reproduction fidelity, differences, and limitations.

## Known Risks

- Paper implementation details may be incomplete.
- NASA `.mat` file structure differs across downloaded subsets.
- Exact train/test split and hyperparameters may require approximation.
- Physics-loss formulation must be checked carefully against the paper before final experiments.
