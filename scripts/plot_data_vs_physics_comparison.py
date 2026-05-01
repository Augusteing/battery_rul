"""Plot true vs data-only vs physics-informed SOH predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--epochs", type=int, default=54)
    parser.add_argument(
        "--battery-ids",
        nargs="+",
        default=["B0005", "B0006", "B0007", "B0018"],
    )
    return parser.parse_args()


def load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("discharge_index").reset_index(drop=True)


def plot_single(
    battery_id: str,
    truth_and_data: pd.DataFrame,
    physics_best: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = truth_and_data.copy()
    merged = merged.rename(columns={"soh_pred": "soh_pred_data_only"})
    merged["soh_pred_physics"] = physics_best["soh_pred"].to_numpy()

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.3, 6.0),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        merged["discharge_index"],
        merged["soh_true"],
        label="True SOH",
        color="#1f4e79",
        linewidth=2.1,
    )
    axes[0].plot(
        merged["discharge_index"],
        merged["soh_pred_data_only"],
        label="Data-only",
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
    )
    axes[0].plot(
        merged["discharge_index"],
        merged["soh_pred_physics"],
        label="Physics-informed (best epoch)",
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
    )
    axes[0].set_ylabel("SOH")
    axes[0].set_title(f"{battery_id} SOH prediction comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].plot(
        merged["discharge_index"],
        merged["soh_pred_data_only"] - merged["soh_true"],
        label="Data-only error",
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
    )
    axes[1].plot(
        merged["discharge_index"],
        merged["soh_pred_physics"] - merged["soh_true"],
        label="Physics error",
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.5,
    )
    axes[1].set_xlabel("Discharge cycle")
    axes[1].set_ylabel("Pred - True")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_overview(predictions: dict[str, tuple[pd.DataFrame, pd.DataFrame]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=180, sharey=False)
    for ax, (battery_id, (data_df, physics_df)) in zip(axes.ravel(), predictions.items()):
        ax.plot(data_df["discharge_index"], data_df["soh_true"], label="True", color="#1f4e79", linewidth=1.8)
        ax.plot(data_df["discharge_index"], data_df["soh_pred"], label="Data-only", color="#d62728", linestyle="--", linewidth=1.6)
        ax.plot(physics_df["discharge_index"], physics_df["soh_pred"], label="Physics-best", color="#2ca02c", linestyle="-.", linewidth=1.6)
        ax.set_title(battery_id)
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH")
        ax.grid(True, alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for battery_id in args.battery_ids:
        data_path = args.tables_dir / f"pi_tnet_data_only_{battery_id}_{args.epochs}epochs_test_predictions.csv"
        physics_path = args.tables_dir / f"pi_tnet_physics_informed_{battery_id}_{args.epochs}epochs_best_test_predictions.csv"
        data_df = load_prediction(data_path)
        physics_df = load_prediction(physics_path)
        predictions[battery_id] = (data_df, physics_df)

        single_path = args.figures_dir / f"comparison_data_vs_physics_{battery_id}_{args.epochs}epochs.png"
        plot_single(battery_id, data_df, physics_df, single_path)
        print(f"Saved: {single_path}")

    overview_path = args.figures_dir / f"comparison_data_vs_physics_{args.epochs}epochs_overview.png"
    plot_overview(predictions, overview_path)
    print(f"Saved: {overview_path}")


if __name__ == "__main__":
    main()
