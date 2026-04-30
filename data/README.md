# Data Directory

`raw/` contains the downloaded NASA battery aging dataset and is treated as read-only.

Recommended data flow:

```text
raw NASA .mat files
  -> data/interim/cycle_records/
  -> data/processed/feature_tables/
  -> data/splits/
```

Do not edit raw files. Any correction, filtering, or feature construction should be implemented in code and recorded in `docs/experiment_log.md`.
