"""Verify PI-TNet NASA training/test data loaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery_rul.data.dataloaders import create_nasa_dataloaders
from battery_rul.data.nasa import PAPER_NASA_CELLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/nasa_pi_tnet_features.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/nasa_pi_tnet_metadata.csv"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def summarize_loader(bundle, battery_id: str | None) -> dict:
    first_train_batch = next(iter(bundle.train))
    first_test_batch = next(iter(bundle.test))
    return {
        "battery_id": battery_id or "all",
        "feature_names": list(bundle.feature_names),
        "train_samples": len(bundle.train.dataset),
        "test_samples": len(bundle.test.dataset),
        "train_batches": len(bundle.train),
        "test_batches": len(bundle.test),
        "batch_size": bundle.train.batch_size,
        "train_shuffle": True,
        "test_shuffle": False,
        "first_train_x_shape": list(first_train_batch["x"].shape),
        "first_test_x_shape": list(first_test_batch["x"].shape),
        "target_keys": ["capacity_ah", "soh"],
    }


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_bundle = create_nasa_dataloaders(
        feature_path=args.features,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    summaries.append(summarize_loader(all_bundle, None))

    for battery_id in PAPER_NASA_CELLS:
        bundle = create_nasa_dataloaders(
            feature_path=args.features,
            metadata_path=args.metadata,
            battery_id=battery_id,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        summaries.append(summarize_loader(bundle, battery_id))

    log = {
        "stage": "M3_dataloader_verification",
        "paper_protocol": {
            "train_split": "first 70% discharge cycles",
            "test_split": "last 30% discharge cycles",
            "train_shuffle": True,
            "test_shuffle": False,
            "batch_size": args.batch_size,
        },
        "summaries": summaries,
    }
    log_path = args.log_dir / "nasa_dataloader_verification.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    for summary in summaries:
        print(
            f"{summary['battery_id']:>5}: "
            f"{summary['train_samples']} train / {summary['test_samples']} test, "
            f"{summary['train_batches']} train batches / {summary['test_batches']} test batches, "
            f"first train batch {summary['first_train_x_shape']}"
        )
    print(f"\nSaved: {log_path}")


if __name__ == "__main__":
    main()
