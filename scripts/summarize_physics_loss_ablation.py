"""Summarize best-epoch physics-loss ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VARIANTS = (
    ("lu_only", "L_u"),
    ("lu_lf", "L_u + L_f"),
    ("lu_lt", "L_u + L_t"),
    ("lu_lf_lt", "L_u + L_f + L_t"),
)


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
        default=Path("results/tables/physics_loss_ablation_best_54epochs.csv"),
    )
    parser.add_argument(
        "--mean-output-path",
        type=Path,
        default=Path("results/tables/physics_loss_ablation_best_54epochs_mean.csv"),
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_run_name(variant_name: str, battery_id: str, epochs: int) -> str:
    return f"physics_ablation_{variant_name}_{battery_id}_{epochs}epochs"


def main() -> None:
    args = parse_args()
    rows = []
    for variant_name, variant_label in VARIANTS:
        for battery_id in args.battery_ids:
            summary_path = (
                args.logs_dir
                / f"{build_run_name(variant_name, battery_id, args.epochs)}_summary.json"
            )
            summary = load_summary(summary_path)
            best = summary["best_epoch_metrics"]
            physics = summary.get("physics", {})
            rows.append(
                {
                    "variant": variant_name,
                    "variant_label": variant_label,
                    "battery_id": battery_id,
                    "best_epoch": summary["best_epoch"],
                    "physics_enabled": summary["physics_enabled"],
                    "use_structural_loss": physics.get("use_structural_loss"),
                    "use_temporal_loss": physics.get("use_temporal_loss"),
                    "test_soh_mae": best["test_soh_mae"],
                    "test_soh_rmse": best["test_soh_rmse"],
                    "test_soh_mape": best["test_soh_mape"],
                    "test_soh_r2": best["test_soh_r2"],
                    "test_capacity_mae": best["test_capacity_mae"],
                    "test_capacity_rmse": best["test_capacity_rmse"],
                    "test_capacity_mape": best["test_capacity_mape"],
                    "test_capacity_r2": best["test_capacity_r2"],
                }
            )

    df = pd.DataFrame(rows).sort_values(["variant", "battery_id"]).reset_index(drop=True)
    mean_df = (
        df.groupby(["variant", "variant_label"], as_index=False)[
            [
                "test_soh_mae",
                "test_soh_rmse",
                "test_soh_mape",
                "test_soh_r2",
                "test_capacity_mae",
                "test_capacity_rmse",
                "test_capacity_mape",
                "test_capacity_r2",
            ]
        ]
        .mean()
        .sort_values("test_soh_rmse")
        .reset_index(drop=True)
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)
    mean_df.to_csv(args.mean_output_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {args.output_path}")
    print(f"Saved: {args.mean_output_path}")


if __name__ == "__main__":
    main()
