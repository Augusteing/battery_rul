# Battery RUL Reproduction

This repository is organized for reproducing the NASA lithium-ion battery experiments in:

> Integrating transformers into physics-informed neural networks: An approach to lithium-ion battery state-of-health prognostics.

The current scope is limited to the NASA battery aging dataset. The repository keeps raw data immutable and separates data processing, model implementation, training, evaluation, and thesis-facing documentation.

## Research Objective

Reproduce a physics-informed neural-network workflow for lithium-ion battery state-of-health estimation and remaining-useful-life prediction on NASA cells, with a focus on reproducibility, transparent experiment records, and thesis-ready figures/tables.

## Repository Layout

```text
battery-rul/
  configs/              # Dataset, model, and experiment YAML files
  data/
    raw/                # Original NASA data, not tracked by git
    interim/            # Parsed cycle-level intermediate files
    processed/          # Model-ready features, labels, and splits
    splits/             # Explicit train/validation/test cell splits
  docs/                 # Thesis notes, reproduction plan, experiment log
  experiments/          # Stable entry points or notebooks for named experiments
  models/               # Checkpoints and exported trained models, not source code
  notebooks/            # Exploratory analysis only
  references/           # Paper notes, citation metadata, formula checks
  results/
    figures/            # Publication/thesis figures
    tables/             # Metrics and ablation tables
    logs/               # Training and evaluation logs
  scripts/              # Command-line scripts for pipeline steps
  src/battery_rul/      # Reusable Python package
```

## Reproduction Stages

1. Validate NASA raw files and extract discharge-cycle capacity/SOH curves.
2. Rebuild the paper's feature set and train/test cell split.
3. Implement a data-driven baseline.
4. Add the physics-informed degradation constraint.
5. Implement the Transformer-based PI-TNet variant.
6. Evaluate SOH/RUL metrics and generate thesis-ready figures.

## Current Status

The NASA-only PI-TNet reproduction pipeline is active on the current branch.

- NASA raw-data audit is complete, including the documented `B0018` cycle-count discrepancy.
- Fixed-length discharge-cycle features are built with a chronological `70/30` split per battery.
- The CDP + Transformer PI-TNet backbone is implemented.
- A data-only baseline has been trained and exported for all four NASA cells.
- The next reproduction step is Verhulst-constrained physics-informed training.
