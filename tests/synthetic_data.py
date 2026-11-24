"""Synthetic stand-ins for data/converted/*.csv and data/raw/*.xlsx.

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
               start: str = "2023-03-30T00:00:00Z", labeled: bool = True,
               temp_by_phase=(12.0, 15.0, 19.0)) -> pd.DataFrame:
    """One sausage. Like the real data, temperature follows the SEASON: it rises over
    series 1 (spring → summer, default) and falls over series 2 (summer → winter), while
    moisture and conductivity follow the ripening in both runs."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    n = int(days * 24 * 60 / interval_min)
    rows = []
    for i in range(n):
        frac = i / n
        phase = 0 if frac < 0.25 else (1 if frac < 0.6 else 2)
        base_t = temp_by_phase[phase]
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
                            start="2023-03-30T00:00:00Z" if series == 1 else "2023-08-01T00:00:00Z",
                            temp_by_phase=(12.0, 15.0, 19.0) if series == 1 else (19.0, 15.0, 11.0))
            p = os.path.join(out_dir, f"series_{series}_sausage_{k}.csv")
            df.to_csv(p, index=False)
            paths.append(p)
    return paths


if __name__ == "__main__":
    import sys
    print(make_dataset(sys.argv[1] if len(sys.argv) > 1 else "/tmp/synthetic_converted"))


def make_raw_workbooks(out_dir: str, n_sausages: int = 2, rows_per_device: int = 30, seed: int = 0) -> list[str]:
    """Excel files with the layout of the real exports (sheet names, headers, device names).

    Layout taken from the saved output of the original conversion notebook. Series 2's
    device names are assumed to follow series 1's pattern.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    t0 = pd.Timestamp("2023-03-30 08:00")
    states = {"light": ["light", "dark"], "pir": ["not triggered", "triggered"]}

    def rows(series):
        out = []
        for i in range(rows_per_device):
            t = t0 + pd.Timedelta(minutes=10 * i)
            phase = PHASES[min(2, i * 3 // rows_per_device)]
            for k in range(1, n_sausages + 1):
                out.append({"t": t, "dev": f"Wurst {k} Feuchtigkeit, Temperatur & Leitwert", "cond": 4000 - i,
                            "moist": 50 - 0.1 * i, "soil_t": 15.0, "phase": phase})
                out.append({"t": t, "dev": f"Wurst {k} Temperatur", "probe_t": 15.5, "phase": phase})
                out.append({"t": t, "dev": f"Wurst {k} PH-Wert", "ph": 5.5, "phase": phase})
            out.append({"t": t, "dev": "Umgebung Temperatur & Luftfeuchtigkeit", "hum": 80.0, "air_t": 14.0, "phase": phase})
            out.append({"t": t, "dev": "Milesight Umgebung PIR & Lichtintensität",
                        "light": states["light"][i % 2], "pir": states["pir"][int(rng.random() < .1)], "phase": phase})
        return out

    s1 = pd.DataFrame([{"Gemessen am": r["t"], "Gerät": r["dev"], "Environment Humdity": r.get("hum"),
                        "Environment Light status": r.get("light"), "Environment Pir status": r.get("pir"),
                        "Sausage conductivity": r.get("cond"), "Sausage moisture": r.get("moist"), "Sausage ph": r.get("ph"),
                        "Sausage Temperature Sensor 1": r.get("air_t"), "Sausage Temperature Sensor 2": r.get("probe_t"),
                        "Sausage Temperature Sensor 3": r.get("soil_t"), "Annotation": r["phase"]} for r in rows(1)])
    s2 = pd.DataFrame([{"Gemessen am": r["t"] + pd.Timedelta(days=60), "Gerät": r["dev"], "Humdity": r.get("hum"),
                        "Light status": r.get("light"), "Pir status": r.get("pir"), "Soil conductivity": r.get("cond"),
                        "Soil moisture": r.get("moist"), "Soil ph": r.get("ph"), "Soil temperature": r.get("soil_t"),
                        "Temp1": r.get("probe_t"), "Temperature2": r.get("air_t")} for r in rows(2)])
    paths = []
    for name, df, sheets in (("series_1.xlsx", s1, ["Filtered", "+ Annotation", "Messwerte Wurst 1, 1. Messreihe"]),
                             ("series_2.xlsx", s2, ["Alle", "Wurst1"])):
        p = os.path.join(out_dir, name)
        with pd.ExcelWriter(p, engine="openpyxl") as xw:
            for sh in sheets:
                (df if sh in ("+ Annotation", "Alle") else df.head(3)).to_excel(xw, sheet_name=sh, index=False)
        paths.append(p)
    return paths
