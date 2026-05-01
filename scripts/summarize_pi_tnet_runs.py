"""Summarize PI-TNet run metrics across batteries."""

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
    parser.add_argument("--output-path", type=Path, default=Path("results/tables/pi_tnet_physics_informed_54epochs_metrics.csv"))
    parser.add_argument(
        "--battery-ids",
        nargs="+",
        default=["B0005", "B0006", "B0007", "B0018"],
    )
    parser.add_argument("--run-prefix", type=str, default="pi_tnet_physics_informed")
    parser.add_argument("--epochs", type=int, default=54)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    rows = []
    for battery_id in args.battery_ids:
        summary_path = args.logs_dir / f"{args.run_prefix}_{battery_id}_{args.epochs}epochs_summary.json"
        summary = load_summary(summary_path)
        final = summary["final_metrics"]
        physics = summary.get("physics", {})
        rows.append(
            {
                "battery_id": battery_id,
                "epochs": summary["epochs"],
                "device": summary["device"],
                "test_soh_mae": final["test_soh_mae"],
                "test_soh_rmse": final["test_soh_rmse"],
                "test_soh_r2": final["test_soh_r2"],
                "test_capacity_mae": final["test_capacity_mae"],
                "test_capacity_rmse": final["test_capacity_rmse"],
                "test_capacity_r2": final["test_capacity_r2"],
                "train_data_loss": final.get("train_data_loss"),
                "train_structural_loss": final.get("train_structural_loss"),
                "train_temporal_loss": final.get("train_temporal_loss"),
                "verhulst_r": physics.get("verhulst_r"),
                "verhulst_R": physics.get("verhulst_R"),
                "verhulst_K": physics.get("verhulst_K"),
                "lambda_u": physics.get("lambda_u"),
                "lambda_t": physics.get("lambda_t"),
                "lambda_f": physics.get("lambda_f"),
            }
        )

    df = pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
