"""Training-only data augmentation.

Why augment: only five annotated sausages from one run are available. The models should not depend on things that
differ between sausages, probes or seasons but say nothing about ripening. Each augmentation adds modified copies of
the *training* data; validation folds and series 2 are never augmented.

Two kinds, chosen by name from AUGMENTATIONS (several can be combined, e.g. ["calibration", "robust"]):

* Row augmentations ("calibration*", "robust") change the raw uplink rows, then the window features are computed
  again. Exact by construction, and they can remove uplinks (which only works on rows). Computed once per recording
  (Windows.with_augmentation); in cross-validation a fold only uses the copies of its training sausages.
* Feature augmentation ("temperature") transforms the window features directly with closed forms (augment_temperature).
  It draws new random values per window, which the row level cannot do because windows overlap.

The magnitudes are judgement calls from the data, not measured probe specifications; notebook 04 compares them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# name → settings. "kind" decides where it is applied; "copies" = augmented copies per recording / window.
#
# calibration*: per probe and per segment of the run (default one day), as if the probe had been calibrated or inserted
#   differently: LSE01 moisture shifted by up to ± moisture_offset percentage points, conductivity multiplied by a
#   factor in conductivity_scale (log-uniform). Targets the differences between sausages (per-sausage scores range
#   0.58–0.81, sausage 1 ≈ 0.60 for every model). Moisture differs by only a few points between the phases in
#   series 1, so too strong a shift removes the signal itself: several strengths to compare. Run of 25 Nov 2026:
#   the random forest gains with strength (0.71 → 0.76 / 0.78 / 0.79 balanced accuracy for small / medium / large),
#   logistic regression and MLP stay within ± 0.01; hence "xl" to find where it stops helping.
# robust: what live operation looks like: measurement noise (Gaussian, σ per signal; conductivity relative) and lost
#   uplinks (each row dropped with probability dropout, which also changes the timing features). Aims at robustness in
#   the case rather than at better scores; the held-out score can only show whether it costs accuracy.
# temperature: per window, temperature level shifted by up to ± offset °C (absolute temperatures only), day/night
#   swings scaled by a factor in scale and mirrored with probability 0.5 (see augment_temperature).
AUGMENTATIONS = {
    "calibration_small": {"kind": "rows", "copies": 2, "segment": "1D",
                          "moisture_offset": 0.5, "conductivity_scale": [0.9, 1 / 0.9]},
    "calibration": {"kind": "rows", "copies": 2, "segment": "1D",
                    "moisture_offset": 1.0, "conductivity_scale": [0.8, 1.25]},
    "calibration_large": {"kind": "rows", "copies": 2, "segment": "1D",
                          "moisture_offset": 2.0, "conductivity_scale": [0.67, 1.5]},
    "calibration_xl": {"kind": "rows", "copies": 2, "segment": "1D",
                       "moisture_offset": 3.0, "conductivity_scale": [0.5, 2.0]},
    "robust": {"kind": "rows", "copies": 2, "dropout": 0.15,
               "noise": {"lse01_moisture": 0.1, "lse01_conductivity": 0.02, "temperature": 0.1}},
    "temperature": {"kind": "features", "copies": 2, "offset": 10.0, "scale": [0.5, 2.0], "mirror": True},
}
# Probe temperatures and their difference to the air: noise on a probe moves both columns (the air stays).
TEMPERATURE_COLUMNS = {"lse01_temperature": "lse01_temperature_diff", "ltcfs_temperature": "ltcfs_temperature_diff"}


def parse(names) -> list[str]:
    """Normalise an augmentation spec: None/"" → [], "a+b" → ["a", "b"], list → list. Unknown names raise."""
    if not names:
        return []
    if isinstance(names, str):
        names = [n for n in names.split("+") if n]
    unknown = [n for n in names if n not in AUGMENTATIONS]
    if unknown:
        raise KeyError(f"unknown augmentation {unknown}; choose from {list(AUGMENTATIONS)}")
    return list(names)


def row_names(names) -> list[str]:
    """The row-level augmentations among `names`."""
    return [n for n in parse(names) if AUGMENTATIONS[n]["kind"] == "rows"]


def feature_names(names) -> list[str]:
    """The feature-level augmentations among `names`."""
    return [n for n in parse(names) if AUGMENTATIONS[n]["kind"] == "features"]


def settings(names) -> dict:
    """The settings of the selected augmentations (for manifest.json and metrics.json)."""
    return {n: AUGMENTATIONS[n] for n in parse(names)}


# ── row level ───────────────────────────────────────────────────────────────
def augment_rows(df: pd.DataFrame, names, rng: np.random.Generator) -> pd.DataFrame:
    """One augmented copy of a recording's rows (columns of data.FEATURE_ORDER, plus "label" if annotated).

    The selected row augmentations are applied one after another to the same copy. Missing values stay missing
    (each row carries the readings of one sensor only).
    """
    out = df.copy()
    for name in row_names(names):
        s = AUGMENTATIONS[name]
        if s.get("dropout"):
            out = out[rng.random(len(out)) >= s["dropout"]]
        if "moisture_offset" in s:
            out = _calibrate(out, s, rng)
        if s.get("noise"):
            out = _add_noise(out, s["noise"], rng)
    return out.reset_index(drop=True)


def _calibrate(df: pd.DataFrame, s: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Per segment of the run: moisture + offset (not below 0), conductivity × factor."""
    t = pd.to_datetime(df["timestamp"])
    seg = ((t - t.iloc[0]) // pd.Timedelta(s["segment"])).to_numpy(dtype=int)
    n = seg.max() + 1 if len(seg) else 0
    offset = rng.uniform(-s["moisture_offset"], s["moisture_offset"], n)
    lo, hi = s["conductivity_scale"]
    factor = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    df["lse01_moisture"] = (df["lse01_moisture"] + offset[seg]).clip(lower=0)
    df["lse01_conductivity"] = df["lse01_conductivity"] * factor[seg]
    return df


def _add_noise(df: pd.DataFrame, noise: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Gaussian measurement noise; conductivity relative to the value, the rest absolute."""
    n = len(df)
    if "lse01_moisture" in noise:
        df["lse01_moisture"] = (df["lse01_moisture"] + rng.normal(0, noise["lse01_moisture"], n)).clip(lower=0)
    if "lse01_conductivity" in noise:
        df["lse01_conductivity"] = df["lse01_conductivity"] * (1 + rng.normal(0, noise["lse01_conductivity"], n))
    if "temperature" in noise:
        for probe, diff in TEMPERATURE_COLUMNS.items():
            e = rng.normal(0, noise["temperature"], n)
            df[probe] = df[probe] + e
            if diff in df:
                df[diff] = df[diff] + e
    return df


def row_copies(names) -> int:
    """Copies per recording: the largest "copies" of the selected row augmentations (they share the copies)."""
    return max((AUGMENTATIONS[n]["copies"] for n in row_names(names)), default=0)


def copy_rng(recording_index: int, copy: int) -> np.random.Generator:
    """Reproducible random numbers per recording and copy, independent of worker order."""
    return np.random.default_rng([RANDOM_STATE, recording_index, copy])


# ── feature level ───────────────────────────────────────────────────────────
def augment_temperature(X: pd.DataFrame, y: np.ndarray, settings: dict | None = None, seed: int = RANDOM_STATE,
                        signal_of=None):
    """Feature-level temperature augmentation. Returns (X, y) with augmented copies added (originals first).

    Per window it applies T' = mean + m·b·(T − mean) + a to every temperature signal: offset a (absolute
    temperatures only, the same for both probes; differences keep their level), swing scale b and mirror m = ±1.
    The window statistics transform in closed form (min/max swap when mirrored, std·b, skew·m, slope·m·b), so this
    works on the features directly; tests/test_ml_core.py checks it against transforming the raw rows.
    signal_of: column → signal name (training.feature_signal; passed in to avoid a circular import).
    """
    if signal_of is None:
        from .training import feature_signal as signal_of
    s = {**AUGMENTATIONS["temperature"], **(settings or {})}
    signals = sorted({signal_of(c) for c in X.columns if signal_of(c) and "temperature" in signal_of(c)})
    if not signals or s["copies"] < 1:
        return X, y
    rng = np.random.default_rng(seed)
    parts, n = [X], len(X)
    for _ in range(s["copies"]):
        a = rng.uniform(-s["offset"], s["offset"], n)
        b = rng.uniform(*s["scale"], n)
        m = np.where(s["mirror"] & (rng.random(n) < 0.5), -1.0, 1.0)
        Xa = X.copy()
        for sig in signals:
            off = 0.0 if sig.endswith("_diff") else a
            mean, lo, hi = X[f"{sig}_mean"], X[f"{sig}_min"], X[f"{sig}_max"]
            Xa[f"{sig}_mean"] = mean + off
            Xa[f"{sig}_min"] = np.where(m > 0, mean + b * (lo - mean), mean - b * (hi - mean)) + off
            Xa[f"{sig}_max"] = np.where(m > 0, mean + b * (hi - mean), mean - b * (lo - mean)) + off
            Xa[f"{sig}_std"] = X[f"{sig}_std"] * b
            Xa[f"{sig}_skew"] = X[f"{sig}_skew"] * m
            Xa[f"{sig}_slope"] = X[f"{sig}_slope"] * m * b
        parts.append(Xa)
    return pd.concat(parts, ignore_index=True), np.tile(y, len(parts))
