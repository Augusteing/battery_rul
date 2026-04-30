# Scripts

Planned command-line entry points:

- `audit_nasa_data.py`: inspect raw NASA files and summarize available cells.
- `build_nasa_dataset.py`: create processed feature tables.
- `build_nasa_features.py`: create fixed-length PI-TNet discharge tensors and chronological 70/30 splits.
- `verify_nasa_dataloader.py`: verify paper-consistent train/test data loaders.
- `verify_pi_tnet_model.py`: verify PI-TNet forward and backward passes on a real NASA batch.
- `train.py`: train baseline or PI-TNet models from YAML configuration.
- `evaluate.py`: compute metrics and export figures.
