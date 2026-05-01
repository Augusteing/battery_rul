"""Plot predicted vs true SOH curves for PI-TNet runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", type=str, default="pi_tnet_data_only")
    parser.add_argument("--epochs", type=int, default=54)
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--battery-ids",
        nargs="+",
        default=["B0005", "B0006", "B0007", "B0018"],
    )
    return parser.parse_args()


def prediction_path(tables_dir: Path, run_prefix: str, battery_id: str, epochs: int) -> Path:
    return tables_dir / f"{run_prefix}_{battery_id}_{epochs}epochs_test_predictions.csv"


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    return df.sort_values("discharge_index").reset_index(drop=True)


def plot_single(df: pd.DataFrame, battery_id: str, output_path: Path) -> None:
    error = df["soh_pred"] - df["soh_true"]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 5.8),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        df["discharge_index"],
        df["soh_true"],
        label="True SOH",
        color="#1f77b4",
        linewidth=2.0,
    )
    axes[0].plot(
        df["discharge_index"],
        df["soh_pred"],
        label="Predicted SOH",
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
    )
    axes[0].set_ylabel("SOH")
    axes[0].set_title(f"{battery_id} data-only PI-TNet test prediction")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].bar(
        df["discharge_index"],
        error,
        color="#6c757d",
        width=0.8,
        alpha=0.85,
    )
    axes[1].set_xlabel("Discharge cycle")
    axes[1].set_ylabel("Pred - True")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_overview(predictions: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=180, sharey=False)
    for ax, (battery_id, df) in zip(axes.ravel(), predictions.items()):
        ax.plot(
            df["discharge_index"],
            df["soh_true"],
            label="True",
            color="#1f77b4",
            linewidth=1.8,
        )
        ax.plot(
            df["discharge_index"],
            df["soh_pred"],
            label="Pred",
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
        )
        ax.set_title(battery_id)
        ax.set_xlabel("Discharge cycle")
        ax.set_ylabel("SOH")
        ax.grid(True, alpha=0.3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for battery_id in args.battery_ids:
        path = prediction_path(args.tables_dir, args.run_prefix, battery_id, args.epochs)
        df = load_prediction(path)
        predictions[battery_id] = df
        single_path = (
            args.figures_dir
            / f"{args.run_prefix}_{battery_id}_{args.epochs}epochs_test_soh.png"
        )
        plot_single(df, battery_id, single_path)
        print(f"Saved: {single_path}")

    overview_path = (
        args.figures_dir / f"{args.run_prefix}_{args.epochs}epochs_test_soh_overview.png"
    )
    plot_overview(predictions, overview_path)
    print(f"Saved: {overview_path}")


if __name__ == "__main__":
    main()
