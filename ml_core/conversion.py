"""Convert the raw Excel exports into one CSV per sausage.

Input: ``data/raw/series_<n>.xlsx``. Each workbook has one sheet with all uplinks
of a curing run ("+ Annotation" in series 1, "Alle" in series 2), one row per
uplink, with the device name in column "Gerät".

Output: ``data/converted/series_<n>_sausage_<k>.csv`` with the columns of
``ml_core.data.ALL_COLUMNS`` (+ ``label`` if the sheet is annotated). Each file
holds the rows of that sausage's two probes plus the shared WS202 rows, because
the models see the uplink stream as it arrived.

Not converted: pH probes and the ambient temperature/humidity device. They are
not part of the demonstrator case.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .data import ALL_COLUMNS

# Device name → sensor type. Checked in this order: LSE01 names also contain "Temperatur".
DEVICE_PATTERNS = [
    ("ws202", re.compile(r"milesight", re.I)),
    ("lse01", re.compile(r"wurst\D*(\d+).*feuchtigkeit", re.I)),
    ("ltcfs", re.compile(r"wurst\D*(\d+).*temperatur", re.I)),
]
# Sheet holding all uplinks of a run.
SHEET_PATTERN = re.compile(r"alle|annotation", re.I)
# Excel column (lower case, spaces → _) → converted column. Series 1 and series 2 use different headers.
COLUMN_MAP = {
    "gemessen_am": "timestamp",
    "gerät": "sensor",
    "annotation": "label",
    "environment_light_status": "ws202_light_status", "light_status": "ws202_light_status",
    "environment_pir_status": "ws202_pir_status", "pir_status": "ws202_pir_status",
    "sausage_conductivity": "lse01_conductivity", "soil_conductivity": "lse01_conductivity",
    "sausage_moisture": "lse01_moisture", "soil_moisture": "lse01_moisture",
    "sausage_temperature_sensor_3": "lse01_temperature", "soil_temperature": "lse01_temperature",
    "sausage_temperature_sensor_2": "ltcfs_temperature", "temp1": "ltcfs_temperature",
}
STATE_MAP = {
    "ws202_light_status": {"dark": 0, "light": 1},
    "ws202_pir_status": {"not triggered": 0, "triggered": 1},
}


def series_number(path: str) -> int:
    m = re.search(r"series_(\d+)", os.path.basename(path), re.I)
    if not m:
        raise ValueError(f"{path}: raw files must be named series_<n>.xlsx")
    return int(m.group(1))


def classify_device(name: str) -> tuple[str | None, int | None]:
    """'Wurst 3 Temperatur' -> ('ltcfs', 3); 'Milesight Umgebung …' -> ('ws202', None)."""
    for sensor, pattern in DEVICE_PATTERNS:
        m = pattern.search(str(name))
        if m:
            return sensor, (int(m.group(1)) if m.groups() else None)
    return None, None


def read_sheet(path: str) -> pd.DataFrame:
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    target = [s for s in sheets if SHEET_PATTERN.search(s)]
    if len(target) != 1:
        raise ValueError(f"{path}: expected exactly one sheet matching 'Alle'/'Annotation', found {target}")
    return pd.read_excel(path, sheet_name=target[0], engine="openpyxl")


def convert_sheet(raw: pd.DataFrame, series: int) -> dict[str, pd.DataFrame]:
    """Raw sheet → {"series_<n>_sausage_<k>": DataFrame}."""
    df = raw.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_")).rename(columns=COLUMN_MAP)
    kinds = df["sensor"].map(classify_device)
    df["sensor"] = [k[0] for k in kinds]
    df["sausage"] = [k[1] for k in kinds]
    df = df[df["sensor"].notna()]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col, mapping in STATE_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v, m=mapping: m.get(str(v).strip().lower(), v))
    columns = ALL_COLUMNS + (["label"] if "label" in df.columns else [])
    df = df.reindex(columns=columns + ["sausage"])

    environment = df[df["sensor"] == "ws202"]
    out = {}
    for k in sorted(df["sausage"].dropna().unique()):
        rows = pd.concat([df[df["sausage"] == k], environment])
        rows = rows.sort_values("timestamp", kind="stable")[columns].reset_index(drop=True)
        out[f"series_{series}_sausage_{int(k)}"] = rows
    return out


def convert_all(raw_dir: str, out_dir: str, log=print) -> pd.DataFrame:
    """Convert every series_<n>.xlsx in raw_dir. Returns a summary (one row per output file)."""
    os.makedirs(out_dir, exist_ok=True)
    summary = []
    files = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(".xlsx") and not f.startswith("~$"))
    for f in files:
        path = os.path.join(raw_dir, f)
        log(f"Reading {f} …")
        for name, rows in convert_sheet(read_sheet(path), series_number(path)).items():
            rows.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
            counts = rows["sensor"].value_counts()
            summary.append({"file": f"{name}.csv", "rows": len(rows),
                            **{f"{s} rows": int(counts.get(s, 0)) for s in ("lse01", "ltcfs", "ws202")},
                            "first": rows["timestamp"].min(), "last": rows["timestamp"].max(),
                            "labeled": int(rows["label"].notna().sum()) if "label" in rows else 0})
            log(f"  wrote {name}.csv ({len(rows):,} rows)")
    return pd.DataFrame(summary)
