"""Shared data conventions for training, replay and live inference.

Everything that must be identical between training time and presentation time
lives here, so the dashboard cannot silently feed the models a different
representation than they were fitted on.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

# Raw-row columns the models consume (same as training notebook cell 2:
# all converted columns minus label and the WS202 environment values).
FEATURE_ORDER = [
    "timestamp",
    "sensor",
    "lse01_conductivity",
    "lse01_moisture",
    "lse01_temperature",
    "ltcfs_temperature",
]

VALUE_COLUMNS = {
    "lse01": ["lse01_conductivity", "lse01_moisture", "lse01_temperature"],
    "ltcfs": ["ltcfs_temperature"],
    "ws202": ["ws202_light_status", "ws202_pir_status"],
}

ALL_COLUMNS = ["timestamp", "sensor"] + [c for cols in VALUE_COLUMNS.values() for c in cols]

# The conversion notebook renames "annotation" -> "label" *before* applying its
# translation map, so converted CSVs keep the German labels. Display both.
LABEL_TRANSLATIONS = {
    "Umroetung": "Curing (Umrötung)",
    "Reifung Phase 1": "Ripening phase 1",
    "Reifung Phase 2": "Ripening phase 2",
    "Curing": "Curing (Umrötung)",
    "Ripening Phase 1": "Ripening phase 1",
    "Ripening Phase 2": "Ripening phase 2",
}

GROUP_RE = re.compile(r"series_(\d+)_sausage_(\d+)")


def group_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_converted_csv(path: str) -> pd.DataFrame:
    """Load one converted recording (one sausage + shared environment rows)."""
    with open(path, "rb") as fh:
        head = fh.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not data. Run `git lfs pull` first."
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def window_starts(n_rows: int, window_size: int, stride: int) -> np.ndarray:
    if n_rows < window_size:
        return np.array([], dtype=int)
    return np.arange(0, n_rows - window_size + 1, stride)


def window_labels(labels: pd.Series, window_size: int, stride: int) -> np.ndarray:
    """Majority label per window (ties -> lexicographically first, as pandas mode)."""
    starts = window_starts(len(labels), window_size, stride)
    vals = labels.to_numpy()
    out = []
    for s in starts:
        out.append(pd.Series(vals[s:s + window_size]).mode().iloc[0])
    return np.array(out, dtype=object)
