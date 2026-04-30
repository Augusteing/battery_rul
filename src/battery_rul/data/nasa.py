"""NASA battery aging dataset readers.

The first reproduction stage follows the NASA subset used by the PI-TNet
paper: B0005, B0006, B0007, and B0018 from BatteryAgingARC-FY08Q4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


PAPER_NASA_CELLS = ("B0005", "B0006", "B0007", "B0018")
NOMINAL_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4


@dataclass(frozen=True)
class DischargeCycle:
    """Cycle-level discharge summary required for SOH/RUL reproduction."""

    battery_id: str
    cycle_index: int
    discharge_index: int
    ambient_temperature: float
    capacity_ah: float
    soh: float
    length: int
    start_time: str
    voltage_min: float
    voltage_max: float
    current_mean: float
    temperature_mean: float
    time_max_s: float


def find_nasa_mat_file(raw_dir: Path, battery_id: str) -> Path:
    """Find one NASA `.mat` file for a battery id under the raw directory."""

    matches = sorted(raw_dir.rglob(f"{battery_id}.mat"))
    if not matches:
        raise FileNotFoundError(f"Cannot find {battery_id}.mat under {raw_dir}")
    if len(matches) > 1:
        fy08q4 = [path for path in matches if "FY08Q4" in str(path)]
        if fy08q4:
            return fy08q4[0]
    return matches[0]


def load_battery_cycles(mat_path: Path, battery_id: str):
    """Load the MATLAB cycle array for a NASA battery."""

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if battery_id not in mat:
        raise KeyError(f"{battery_id} is not a top-level key in {mat_path}")
    return mat[battery_id].cycle


def _to_1d_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _matlab_time_to_string(value) -> str:
    values = np.asarray(value).reshape(-1)
    if values.size < 6:
        return ""
    year, month, day, hour, minute = values[:5].astype(int)
    second = float(values[5])
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:05.2f}"


def extract_discharge_cycles(
    raw_dir: Path,
    battery_id: str,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> list[DischargeCycle]:
    """Extract discharge cycles and SOH labels from one NASA battery."""

    mat_path = find_nasa_mat_file(raw_dir, battery_id)
    cycles = load_battery_cycles(mat_path, battery_id)
    discharge_cycles: list[DischargeCycle] = []

    for cycle_index, cycle in enumerate(cycles, start=1):
        if str(cycle.type).lower() != "discharge":
            continue

        data = cycle.data
        voltage = _to_1d_array(data.Voltage_measured)
        current = _to_1d_array(data.Current_measured)
        temperature = _to_1d_array(data.Temperature_measured)
        time_s = _to_1d_array(data.Time)
        capacity_ah = float(np.asarray(data.Capacity).reshape(-1)[0])
        discharge_index = len(discharge_cycles) + 1

        discharge_cycles.append(
            DischargeCycle(
                battery_id=battery_id,
                cycle_index=cycle_index,
                discharge_index=discharge_index,
                ambient_temperature=float(cycle.ambient_temperature),
                capacity_ah=capacity_ah,
                soh=capacity_ah / nominal_capacity_ah,
                length=int(len(time_s)),
                start_time=_matlab_time_to_string(cycle.time),
                voltage_min=float(np.min(voltage)),
                voltage_max=float(np.max(voltage)),
                current_mean=float(np.mean(current)),
                temperature_mean=float(np.mean(temperature)),
                time_max_s=float(np.max(time_s)),
            )
        )

    return discharge_cycles


def extract_discharge_curve_records(
    raw_dir: Path,
    battery_id: str,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> list[dict]:
    """Extract raw discharge curves for one NASA battery.

    Each record contains the paper-required input variables:
    voltage, current, temperature, and time, plus capacity/SOH labels.
    """

    mat_path = find_nasa_mat_file(raw_dir, battery_id)
    cycles = load_battery_cycles(mat_path, battery_id)
    records: list[dict] = []

    for cycle_index, cycle in enumerate(cycles, start=1):
        if str(cycle.type).lower() != "discharge":
            continue

        data = cycle.data
        capacity_ah = float(np.asarray(data.Capacity).reshape(-1)[0])
        records.append(
            {
                "battery_id": battery_id,
                "cycle_index": cycle_index,
                "discharge_index": len(records) + 1,
                "ambient_temperature": float(cycle.ambient_temperature),
                "start_time": _matlab_time_to_string(cycle.time),
                "capacity_ah": capacity_ah,
                "soh": capacity_ah / nominal_capacity_ah,
                "voltage_measured": _to_1d_array(data.Voltage_measured),
                "current_measured": _to_1d_array(data.Current_measured),
                "temperature_measured": _to_1d_array(data.Temperature_measured),
                "time": _to_1d_array(data.Time),
            }
        )

    return records


def interpolate_discharge_record(
    record: dict,
    points_per_cycle: int,
    feature_names: tuple[str, ...] = (
        "voltage_measured",
        "current_measured",
        "temperature_measured",
        "time",
    ),
) -> np.ndarray:
    """Interpolate one discharge record to a fixed normalized time grid.

    NASA discharge curves have variable sampling lengths. The PI-TNet paper
    states that voltage/current values are compared at identical sampling time
    points, but does not disclose the exact number of points. We therefore
    construct a reproducible normalized discharge-time grid.
    """

    source_time = _to_1d_array(record["time"])
    if source_time.size < 2:
        raise ValueError(
            f"{record['battery_id']} discharge {record['discharge_index']} "
            "has fewer than two time samples"
        )
    denominator = source_time[-1] - source_time[0]
    if denominator <= 0:
        raise ValueError(
            f"{record['battery_id']} discharge {record['discharge_index']} "
            "has non-increasing time samples"
        )

    source_grid = (source_time - source_time[0]) / denominator
    target_grid = np.linspace(0.0, 1.0, points_per_cycle)
    channels = []
    for feature_name in feature_names:
        values = _to_1d_array(record[feature_name])
        if values.size != source_grid.size:
            raise ValueError(
                f"{record['battery_id']} discharge {record['discharge_index']} "
                f"feature {feature_name} length does not match time length"
            )
        channels.append(np.interp(target_grid, source_grid, values))
    return np.stack(channels, axis=0).astype(np.float32)


def build_interpolated_feature_dataset(
    raw_dir: Path,
    battery_ids: Iterable[str] = PAPER_NASA_CELLS,
    points_per_cycle: int = 128,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create fixed-length discharge tensors and aligned metadata.

    Returns
    -------
    features:
        Array with shape `(n_cycles, 4, points_per_cycle)` for V/I/T/t.
    metadata:
        One row per discharge cycle with capacity and SOH labels.
    """

    feature_arrays = []
    metadata_rows = []
    for battery_id in battery_ids:
        for record in extract_discharge_curve_records(
            raw_dir=raw_dir,
            battery_id=battery_id,
            nominal_capacity_ah=nominal_capacity_ah,
        ):
            feature_arrays.append(
                interpolate_discharge_record(record, points_per_cycle)
            )
            metadata_rows.append(
                {
                    "battery_id": record["battery_id"],
                    "cycle_index": record["cycle_index"],
                    "discharge_index": record["discharge_index"],
                    "ambient_temperature": record["ambient_temperature"],
                    "start_time": record["start_time"],
                    "capacity_ah": record["capacity_ah"],
                    "soh": record["soh"],
                    "source_length": int(len(record["time"])),
                    "duration_s": float(record["time"][-1]),
                }
            )

    return np.stack(feature_arrays, axis=0), pd.DataFrame(metadata_rows)


def assign_chronological_70_30_split(metadata: pd.DataFrame) -> pd.DataFrame:
    """Assign the paper's first-70%-train and last-30%-test split per cell."""

    split = metadata.copy()
    split["split"] = "test"
    for battery_id, index in split.groupby("battery_id", sort=True).groups.items():
        cell_rows = split.loc[index].sort_values("discharge_index")
        train_count = int(np.floor(len(cell_rows) * 0.70))
        train_indices = cell_rows.iloc[:train_count].index
        split.loc[train_indices, "split"] = "train"
    return split


def build_discharge_summary(
    raw_dir: Path,
    battery_ids: Iterable[str] = PAPER_NASA_CELLS,
    nominal_capacity_ah: float = NOMINAL_CAPACITY_AH,
) -> pd.DataFrame:
    """Build a tidy discharge-cycle summary for the paper's NASA cells."""

    rows = []
    for battery_id in battery_ids:
        rows.extend(extract_discharge_cycles(raw_dir, battery_id, nominal_capacity_ah))
    return pd.DataFrame([cycle.__dict__ for cycle in rows])


def build_cell_audit_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize cell-level facts needed before feature construction."""

    records = []
    for battery_id, group in summary.groupby("battery_id", sort=True):
        group = group.sort_values("discharge_index")
        eol_rows = group[group["capacity_ah"] <= EOL_CAPACITY_AH]
        first_eol_cycle = (
            int(eol_rows.iloc[0]["discharge_index"]) if not eol_rows.empty else np.nan
        )
        records.append(
            {
                "battery_id": battery_id,
                "discharge_cycles": int(len(group)),
                "first_capacity_ah": float(group.iloc[0]["capacity_ah"]),
                "last_capacity_ah": float(group.iloc[-1]["capacity_ah"]),
                "min_capacity_ah": float(group["capacity_ah"].min()),
                "max_capacity_ah": float(group["capacity_ah"].max()),
                "first_soh": float(group.iloc[0]["soh"]),
                "last_soh": float(group.iloc[-1]["soh"]),
                "first_eol_discharge_cycle": first_eol_cycle,
                "min_curve_length": int(group["length"].min()),
                "max_curve_length": int(group["length"].max()),
            }
        )
    return pd.DataFrame.from_records(records)
