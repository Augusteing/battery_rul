"""Run physics-loss ablation experiments for PI-TNet."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_pi_tnet.py"
SUMMARIZE_SCRIPT = PROJECT_ROOT / "scripts" / "summarize_physics_loss_ablation.py"
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plot_physics_loss_ablation.py"

VARIANTS = (
    {
        "name": "lu_only",
        "label": "L_u",
        "extra_args": ["--no-physics"],
    },
    {
        "name": "lu_lf",
        "label": "L_u + L_f",
        "extra_args": ["--disable-temporal-loss"],
    },
    {
        "name": "lu_lt",
        "label": "L_u + L_t",
        "extra_args": ["--disable-structural-loss"],
    },
    {
        "name": "lu_lf_lt",
        "label": "L_u + L_f + L_t",
        "extra_args": [],
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model/pi_tnet.yaml"))
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
    parser.add_argument(
        "--battery-ids",
        nargs="+",
        default=["B0005", "B0006", "B0007", "B0018"],
    )
    parser.add_argument("--epochs", type=int, default=54)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def summary_exists(output_dir: Path, run_name: str) -> bool:
    return (output_dir / "logs" / f"{run_name}_summary.json").exists()


def main() -> None:
    args = parse_args()
    for variant in VARIANTS:
        for battery_id in args.battery_ids:
            run_name = f"physics_ablation_{variant['name']}_{battery_id}_{args.epochs}epochs"
            if args.skip_existing and summary_exists(args.output_dir, run_name):
                print(f"Skipping existing run: {run_name}")
                continue
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(args.config),
                "--features",
                str(args.features),
                "--metadata",
                str(args.metadata),
                "--battery-id",
                battery_id,
                "--run-name",
                run_name,
                "--output-dir",
                str(args.output_dir),
                "--max-epochs",
                str(args.epochs),
                "--device",
                args.device,
                "--physics-time-mode",
                "raw",
                "--physics-data-loss-mode",
                "capacity_loss",
                *variant["extra_args"],
            ]
            run_command(command)

    run_command(
        [
            sys.executable,
            str(SUMMARIZE_SCRIPT),
            "--logs-dir",
            str(args.output_dir / "logs"),
            "--epochs",
            str(args.epochs),
            "--output-path",
            str(args.output_dir / "tables" / f"physics_loss_ablation_best_{args.epochs}epochs.csv"),
            "--mean-output-path",
            str(
                args.output_dir
                / "tables"
                / f"physics_loss_ablation_best_{args.epochs}epochs_mean.csv"
            ),
            "--battery-ids",
            *args.battery_ids,
        ]
    )
    run_command(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input-path",
            str(args.output_dir / "tables" / f"physics_loss_ablation_best_{args.epochs}epochs.csv"),
            "--output-path",
            str(
                args.output_dir
                / "figures"
                / f"physics_loss_ablation_best_{args.epochs}epochs_soh_rmse.png"
            ),
        ]
    )


if __name__ == "__main__":
    main()
