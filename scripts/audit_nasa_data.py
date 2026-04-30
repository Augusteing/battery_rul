"""Audit the NASA cells used in the PI-TNet reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from battery_rul.data.nasa import (
    EOL_CAPACITY_AH,
    NOMINAL_CAPACITY_AH,
    PAPER_NASA_CELLS,
    build_cell_audit_table,
    build_discharge_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def save_capacity_figure(summary, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    for battery_id in PAPER_NASA_CELLS:
        group = summary[summary["battery_id"] == battery_id].sort_values(
            "discharge_index"
        )
        ax.plot(
            group["discharge_index"],
            group["capacity_ah"],
            linewidth=1.8,
            label=battery_id,
        )

    ax.axhline(
        EOL_CAPACITY_AH,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="EOL = 1.4 Ah",
    )
    ax.set_xlabel("Discharge cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("NASA battery capacity degradation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    logs_dir = args.output_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    summary = build_discharge_summary(
        raw_dir=args.raw_dir,
        battery_ids=PAPER_NASA_CELLS,
        nominal_capacity_ah=NOMINAL_CAPACITY_AH,
    )
    audit = build_cell_audit_table(summary)

    summary_path = args.processed_dir / "nasa_discharge_summary.csv"
    audit_path = tables_dir / "nasa_cell_audit.csv"
    figure_path = figures_dir / "nasa_capacity_degradation.png"
    log_path = logs_dir / "nasa_data_audit.json"

    summary.to_csv(summary_path, index=False)
    audit.to_csv(audit_path, index=False)
    save_capacity_figure(summary, figure_path)

    log = {
        "stage": "M1_data_audit",
        "paper_cells": list(PAPER_NASA_CELLS),
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "eol_capacity_ah": EOL_CAPACITY_AH,
        "raw_dir": str(args.raw_dir),
        "outputs": {
            "discharge_summary": str(summary_path),
            "cell_audit": str(audit_path),
            "capacity_figure": str(figure_path),
        },
        "audit": audit.to_dict(orient="records"),
    }
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(audit.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {audit_path}")
    print(f"Saved: {figure_path}")
    print(f"Saved: {log_path}")


if __name__ == "__main__":
    main()
