"""Compare data-only final metrics against physics-informed best-epoch metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("results/logs"))
    parser.add_argument(
        "--battery-ids",
        nargs="+",
        default=["B0005", "B0006", "B0007", "B0018"],
    )
    parser.add_argument("--epochs", type=int, default=54)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/tables/pi_tnet_data_vs_best_physics_54epochs_comparison.csv"),
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    rows = []
    for battery_id in args.battery_ids:
        data_summary = load_summary(
            args.logs_dir / f"pi_tnet_data_only_{battery_id}_{args.epochs}epochs_summary.json"
        )
        physics_summary = load_summary(
            args.logs_dir / f"pi_tnet_physics_informed_{battery_id}_{args.epochs}epochs_summary.json"
        )
        data_final = data_summary["final_metrics"]
        physics_best = physics_summary["best_epoch_metrics"]
        rows.append(
            {
                "battery_id": battery_id,
                "physics_best_epoch": physics_summary["best_epoch"],
                "data_only_mae": data_final["test_soh_mae"],
                "physics_best_mae": physics_best["test_soh_mae"],
                "mae_delta": physics_best["test_soh_mae"] - data_final["test_soh_mae"],
                "data_only_rmse": data_final["test_soh_rmse"],
                "physics_best_rmse": physics_best["test_soh_rmse"],
                "rmse_delta": physics_best["test_soh_rmse"] - data_final["test_soh_rmse"],
                "data_only_r2": data_final["test_soh_r2"],
                "physics_best_r2": physics_best["test_soh_r2"],
                "r2_delta": physics_best["test_soh_r2"] - data_final["test_soh_r2"],
            }
        )

    df = pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
