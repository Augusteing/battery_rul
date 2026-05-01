"""Plot grouped SOH RMSE results for physics-loss ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


VARIANT_ORDER = ["L_u", "L_u + L_f", "L_u + L_t", "L_u + L_f + L_t"]
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("results/tables/physics_loss_ablation_best_54epochs.csv"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("results/figures/physics_loss_ablation_best_54epochs_soh_rmse.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_path)
    pivot = df.pivot(index="battery_id", columns="variant_label", values="test_soh_rmse")
    pivot = pivot[VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    x = range(len(pivot.index))
    width = 0.18
    for i, variant in enumerate(VARIANT_ORDER):
        positions = [value + (i - 1.5) * width for value in x]
        values = pivot[variant].to_numpy()
        ax.bar(positions, values, width=width, label=variant, color=COLORS[i])

    ax.set_xticks(list(x))
    ax.set_xticklabels(list(pivot.index))
    ax.set_ylabel("SOH RMSE")
    ax.set_title("Physics-loss ablation on NASA batteries")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
