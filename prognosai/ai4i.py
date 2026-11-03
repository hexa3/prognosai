"""Machine-failure data loading + physics-informed feature engineering (AI4I 2020).

The AI4I 2020 dataset (UCI #601) describes 10,000 milling operations with five
failure mechanisms. Each mechanism comes from a physical cause, so we turn those
causes into features instead of leaving the model to find them unaided:

* Heat dissipation failure (HDF): the gap between process and air temperature
  becomes too small while the spindle runs slowly.
* Power failure (PWF): mechanical power = torque × angular velocity leaves the
  safe band.
* Overstrain failure (OSF): tool wear × torque goes over a limit that depends
  on product quality.
* Tool wear failure (TWF): the tool approaches its end of life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import DATA_DIR

RAW_PATH = DATA_DIR / "ai4i" / "ai4i2020.csv"

RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
    "Type": "quality",
}

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF"]  # RNF is random by construction -> not learnable
MODE_NAMES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
}

RAW_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min", "quality_code"]
ENGINEERED_FEATURES = [
    "temp_diff_k",
    "power_w",
    "strain_nm_min",
    "strain_ratio",
    "power_margin_w",
    "wear_ratio",
]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

QUALITY_CODE = {"L": 0, "M": 1, "H": 2}
# Overstrain limits by product quality (minNm).
OSF_LIMIT = {0: 11_000.0, 1: 12_000.0, 2: 13_000.0}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, encoding="utf-8-sig").rename(columns=RENAME)
    df["quality_code"] = df["quality"].map(QUALITY_CODE).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed features. Works for a single row or a full frame."""
    out = df.copy()
    if "quality_code" not in out and "quality" in out:
        out["quality_code"] = out["quality"].map(QUALITY_CODE).astype(int)
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60.0
    out["strain_nm_min"] = out["tool_wear_min"] * out["torque_nm"]
    limit = out["quality_code"].map(OSF_LIMIT).astype(float)
    out["strain_ratio"] = out["strain_nm_min"] / limit
    # Distance from the closest edge of the safe power band (3.5 kW to 9 kW); negative means outside it.
    out["power_margin_w"] = np.minimum(out["power_w"] - 3500.0, 9000.0 - out["power_w"])
    out["wear_ratio"] = out["tool_wear_min"] / 240.0
    return out


def load_dataset() -> pd.DataFrame:
    return add_features(load_raw())
