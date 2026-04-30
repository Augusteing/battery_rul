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
