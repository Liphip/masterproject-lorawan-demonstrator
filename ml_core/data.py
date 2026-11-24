"""Paths and data conventions shared by conversion, training, replay and live inference.

Everything that must be identical at training time and presentation time lives
here, so the demonstrator cannot feed the models a different representation
than they were fitted on.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")              # series_<n>.xlsx exports
CONVERTED_DIR = os.path.join(DATA_DIR, "converted")  # series_<n>_sausage_<k>.csv
MODELS_DIR = os.path.join(DATA_DIR, "models")        # trained pipelines, metrics, hyperparameters
RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")  # live uplinks (not committed)

# Raw-row columns the models consume: everything except the label and the WS202 values.
FEATURE_ORDER = [
    "timestamp",
    "sensor",
    "lse01_conductivity",
    "lse01_moisture",
    "lse01_temperature",
    "ltcfs_temperature",
]
# Annotated curing phases in chronological order (labels as in the raw data).
PHASES = ["Umroetung", "Reifung Phase 1", "Reifung Phase 2"]
# All value columns of a converted recording. WS202 (PIR/light) rows are kept because
# they are part of the uplink stream the windows are built from, but not as features.
ALL_COLUMNS = FEATURE_ORDER + ["ws202_light_status", "ws202_pir_status"]


def recording_name(path: str) -> str:
    """data/converted/series_1_sausage_2.csv -> series_1_sausage_2"""
    return os.path.splitext(os.path.basename(path))[0]


def is_lfs_pointer(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(40).startswith(b"version https://git-lfs")


def load_recording(path: str) -> pd.DataFrame:
    """Load one converted recording (one sausage plus the shared environment rows)."""
    if is_lfs_pointer(path):
        raise RuntimeError(f"{path} is a Git LFS pointer, not data. Run `git lfs pull` first.")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").astype("datetime64[ns, UTC]")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def in_phase_order(labels) -> list:
    """Sort labels chronologically (unknown labels last, alphabetically)."""
    return sorted(set(labels), key=lambda c: (PHASES.index(c) if c in PHASES else len(PHASES), str(c)))


def window_labels(labels: pd.Series, window_size: int, stride: int) -> np.ndarray:
    """Majority label per window, aligned with WindowFeatures (ties: first in sort order)."""
    vals = labels.to_numpy()
    starts = range(0, len(vals) - window_size + 1, stride) if len(vals) >= window_size else []
    return np.array([pd.Series(vals[s:s + window_size]).mode().iloc[0] for s in starts], dtype=object)
