"""Build fixed-length NASA discharge features for PI-TNet reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from battery_rul.data.nasa import (
    NOMINAL_CAPACITY_AH,
    PAPER_NASA_CELLS,
    assign_chronological_70_30_split,
    build_interpolated_feature_dataset,
)


FEATURE_NAMES = (
    "voltage_measured",
    "current_measured",
    "temperature_measured",
    "time",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--points-per-cycle", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    features, metadata = build_interpolated_feature_dataset(
        raw_dir=args.raw_dir,
        battery_ids=PAPER_NASA_CELLS,
        points_per_cycle=args.points_per_cycle,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
    )
    metadata = assign_chronological_70_30_split(metadata)

    capacity = metadata["capacity_ah"].to_numpy(dtype=np.float32)
    soh = metadata["soh"].to_numpy(dtype=np.float32)
    battery_id = metadata["battery_id"].to_numpy(dtype=str)
    discharge_index = metadata["discharge_index"].to_numpy(dtype=np.int32)

    feature_path = args.processed_dir / "nasa_pi_tnet_features.npz"
    metadata_path = args.processed_dir / "nasa_pi_tnet_metadata.csv"
    split_path = args.split_dir / "nasa_chronological_70_30_split.csv"
    log_path = args.log_dir / "nasa_feature_build.json"

    np.savez_compressed(
        feature_path,
        x=features,
        capacity_ah=capacity,
        soh=soh,
        battery_id=battery_id,
        discharge_index=discharge_index,
        feature_names=np.asarray(FEATURE_NAMES, dtype=str),
    )
    metadata.to_csv(metadata_path, index=False)
    metadata[
        ["battery_id", "discharge_index", "cycle_index", "split", "capacity_ah", "soh"]
    ].to_csv(split_path, index=False)

    split_counts = (
        metadata.groupby(["battery_id", "split"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    log = {
        "stage": "M2_feature_construction",
        "paper_cells": list(PAPER_NASA_CELLS),
        "feature_names": list(FEATURE_NAMES),
        "feature_shape": list(features.shape),
        "points_per_cycle": args.points_per_cycle,
        "label_capacity_unit": "Ah",
        "soh_definition": "capacity_ah / 2.0",
        "split_protocol": "first 70% discharge cycles for train, last 30% for test, per cell",
        "normalization": "none; raw physical units retained",
        "interpolation_note": (
            "NASA discharge curves have variable sample lengths. The paper "
            "requires identical sampling points but does not disclose the "
            "exact count; fixed-length normalized-time interpolation is used."
        ),
        "outputs": {
            "features": str(feature_path),
            "metadata": str(metadata_path),
            "split": str(split_path),
        },
        "split_counts": split_counts.to_dict(orient="records"),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"Feature tensor shape: {features.shape}")
    print(split_counts.to_string(index=False))
    print(f"\nSaved: {feature_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved: {split_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
