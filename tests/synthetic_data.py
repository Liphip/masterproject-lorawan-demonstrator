"""Synthetic stand-in for data/training/converted/*.csv.

TEST USE ONLY. The values are invented and only mimic the *schema* and rough
shape of the real "Ahle Wurst" recordings (columns, sensor names, German phase
labels, interleaved uplinks). Never use this for evaluation or presentation.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

PHASES = ["Umroetung", "Reifung Phase 1", "Reifung Phase 2"]
COLUMNS = [
    "timestamp", "sensor",
    "lse01_conductivity", "lse01_moisture", "lse01_temperature", "ltcfs_temperature",
    "ws202_light_status", "ws202_pir_status", "label",
]


def make_group(seed: int, days: float = 4.0, interval_min: float = 10.0,
               start: str = "2023-03-30T00:00:00Z", labeled: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    n = int(days * 24 * 60 / interval_min)
    rows = []
    for i in range(n):
        frac = i / n
        phase = 0 if frac < 0.25 else (1 if frac < 0.6 else 2)
        base_t = [18.0, 14.0, 12.0][phase]
        ts = t0 + pd.Timedelta(minutes=i * interval_min)
        jitter = lambda: pd.Timedelta(seconds=float(rng.uniform(0, 90)))
        moisture = 60 - 35 * frac + rng.normal(0, 1.5) + seed % 3
        cond = 5000 - 3000 * frac + rng.normal(0, 150)
        rows.append({"timestamp": ts + jitter(), "sensor": "lse01",
                     "lse01_conductivity": cond, "lse01_moisture": moisture,
                     "lse01_temperature": base_t + rng.normal(0, 0.3),
                     "label": PHASES[phase]})
        rows.append({"timestamp": ts + jitter(), "sensor": "ltcfs",
                     "ltcfs_temperature": base_t + 0.4 + rng.normal(0, 0.3),
                     "label": PHASES[phase]})
        if i % 2 == 0:
            rows.append({"timestamp": ts + jitter(), "sensor": "ws202",
                         "ws202_light_status": float(rng.random() < 0.3),
                         "ws202_pir_status": float(rng.random() < 0.05),
                         "label": PHASES[phase]})
    df = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if not labeled:
        df = df.drop(columns=["label"])
    return df.reset_index(drop=True)


def make_dataset(out_dir: str, n_sausages: int = 5, days: float = 4.0, seed: int = 0) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for series, labeled in ((1, True), (2, False)):
        for k in range(1, n_sausages + 1):
            df = make_group(seed + 10 * series + k, days=days, labeled=labeled,
                            start="2023-03-30T00:00:00Z" if series == 1 else "2023-06-01T00:00:00Z")
            p = os.path.join(out_dir, f"series_{series}_sausage_{k}.csv")
            df.to_csv(p, index=False)
            paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    print(make_dataset(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_converted"))
