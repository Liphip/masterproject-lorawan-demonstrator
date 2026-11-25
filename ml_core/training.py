"""Hyperparameter search, leave-one-sausage-out evaluation and model export.

Used by notebooks 02 (search), 03 (train and export) and 04 (seasonal confound check).
Headless alternative for the training step:  python -m ml_core.training

Feature sets: the only annotated run (series 1) goes from spring to summer, so temperature
rises together with the ripening phase and models can learn the season instead of the ripening.
FEATURE_SETS selects which sensor signals the models may use. temperature_difference uses the
probe temperatures relative to the air temperature, which removes the seasonal level. Optional
augmentation (augment_temperature) additionally randomises temperature level and swings in the
training data. The unannotated series 2 (summer to spring) serves as a label-free check: a model
that learned ripening must predict phases that advance in time there too (predict_progression,
progression_scores).

Output in data/models/ (what the demonstrator loads):
    hyperparameters.json   search result (written by the search, read by training)
    manifest.json          model registry, window settings, feature schema and ranges, library versions
    metrics.json           per-fold scores, confusion matrices, latency, size, model agreement,
                           phase progression on the unannotated run
    final/                 models fitted on all labeled sausages
    folds/<recording>/     models fitted WITHOUT that sausage (honest replay)
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import importlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, replace

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

importlib.import_module("sklearn.experimental.enable_halving_search_cv")  # makes HalvingGridSearchCV importable
from sklearn.model_selection import HalvingGridSearchCV, LeaveOneGroupOut  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import (CONVERTED_DIR, FEATURE_ORDER, MODELS_DIR, in_phase_order, load_recording, recording_name,
                   window_labels)
from .pipeline import ModelPipeline
from .windowing import WindowFeatures

# Windows in which a sensor never reported give all-NaN feature columns in some folds.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

RANDOM_STATE = 42
# Parallel worker processes for windowing, cross-validation folds and feature-set comparison (-1 = all cores).
# Each worker fits single-threaded, so there is no oversubscription. Set ML_N_JOBS to override (e.g. 4 if RAM is short).
N_JOBS = int(os.environ.get("ML_N_JOBS", "-1"))
# Workers get their data pickled instead of memory-mapped from temp folders: on Windows, loky's resource tracker
# otherwise prints harmless but alarming "KeyError: ...joblib_memmapping_folder..." tracebacks when it cleans up.
# Costs one copy of the training windows per worker (tens of MB).
NO_MEMMAP = {"max_nbytes": None}
MODEL_NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "mlp": "MLP (neural net)"}
HYPERPARAMETERS_FILE = "hyperparameters.json"

# Used when no search result exists: winners of the first search (Nov 2026, before
# the slope-feature fix). Re-run notebook 02 to replace them.
DEFAULT_PARAMS = {
    "logreg": {"solver": "liblinear", "C": 100, "l1_ratio": 0, "class_weight": None, "max_iter": 1000},
    "rf": {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 4,
           "max_features": "sqrt", "bootstrap": True, "class_weight": "balanced_subsample", "criterion": "gini"},
    "mlp": {"hidden_layer_sizes": [64, 32, 16], "activation": "relu", "alpha": 1e-4, "learning_rate_init": 1e-3,
            "batch_size": 32, "max_iter": 500, "early_stopping": True, "validation_fraction": 0.15},
}
# Search space. Logistic regression: l1_ratio 0 = L2, 1 = L1, in between = elastic net.
PARAM_GRIDS = {
    "logreg": [
        {"solver": ["liblinear"], "l1_ratio": [0, 1], "C": [0.01, 1, 100], "class_weight": [None, "balanced"], "max_iter": [1000]},
        {"solver": ["lbfgs"], "l1_ratio": [0], "C": [0.01, 1, 10], "class_weight": [None, "balanced"], "max_iter": [1000]},
        {"solver": ["saga"], "l1_ratio": [0.2, 0.5, 0.8], "C": [0.1, 1, 10], "class_weight": ["balanced"], "max_iter": [1000]},
    ],
    "rf": {"n_estimators": [200, 500], "max_depth": [None, 10], "min_samples_split": [2], "min_samples_leaf": [1, 4],
           "max_features": ["sqrt", 0.7], "bootstrap": [True], "class_weight": [None, "balanced_subsample"], "criterion": ["gini"]},
    "mlp": {"hidden_layer_sizes": [[32], [32, 16], [64, 32], [64, 32, 16]], "activation": ["relu", "tanh"],
            "alpha": [1e-4, 1e-3], "learning_rate_init": [3e-4, 1e-3], "batch_size": [32], "max_iter": [500],
            "early_stopping": [True], "validation_fraction": [0.15]},
}
# A forest with unlimited depth on ~160k windows grows to hundreds of MB per model.
# Capped for the demo laptop; pass limit_size=False to export the searched forest as is.
SIZE_LIMITS = {"rf": {"n_estimators": 100, "max_depth": 10}}
# Sensor signals the models may use.
#   original               the first feature set: absolute temperatures and uplink timing included
#   no_temperature         without temperature
#   moisture_conductivity  without temperature and uplink timing (timing carries no ripening information)
#   temperature_difference moisture, conductivity and probe temperature minus air temperature:
#                          the seasonal level cancels out, the sausage's own behaviour (e.g. evaporative
#                          cooling while it is moist) remains
FEATURE_SETS = {
    "original": ["lse01_conductivity", "lse01_moisture", "lse01_temperature", "ltcfs_temperature", "timestamp"],
    "no_temperature": ["lse01_conductivity", "lse01_moisture", "timestamp"],
    "moisture_conductivity": ["lse01_conductivity", "lse01_moisture"],
    "temperature_difference": ["lse01_conductivity", "lse01_moisture", "lse01_temperature_diff", "ltcfs_temperature_diff"],
}
DEFAULT_FEATURE_SET = "temperature_difference"
# Statistics WindowFeatures computes per signal (used to select feature columns exactly).
SIGNAL_STATS = {"min", "max", "mean", "std", "skew", "slope"}
TIMING_STATS = {"count", "duration", "interval", "min_interval", "max_interval"}
# Training-only augmentation of temperature features, per window (see augment_temperature):
#   offset  ± °C shift of the absolute temperature level (the same for both probes)
#   scale   factor range for the swings around the window mean (day/night amplitude is seasonal)
#   mirror  flip the swings (rising ↔ falling) with probability 0.5
#   copies  augmented copies per training window (the original is kept)
AUGMENTATION = {"offset": 10.0, "scale": [0.5, 2.0], "mirror": True, "copies": 2}
# Smoothing of predicted phases before scoring their progression (one day of windows).
PROGRESSION_SMOOTHING = "1D"
# Not scored at the start of a run: in series 2 every model predicts Phase 2 for the first 1–3 days, probably while the
# probes were not yet (fully) in the sausage.
PROGRESSION_WARMUP = "3D"
# Backwards share counts only regressions that last: the phase is smoothed again over this period first.
PROGRESSION_SUSTAINED = "7D"
# Series 2: from early February 2024 LSE01 moisture jumps from ~7 % to ~22 % and every model falls back to earlier
# phases. Probably the probes were moved or re-inserted (no log exists). The check only scores the time before this.
PROGRESSION_UNTIL = "2024-02-01"
# Tiny settings for smoke tests (notebook QUICK flag, test suite).
QUICK_PARAMS = {"rf": {"n_estimators": 20}, "mlp": {"max_iter": 30, "batch_size": 128}}
QUICK_GRIDS = {"logreg": {"solver": ["lbfgs"], "l1_ratio": [0], "C": [0.1, 10], "max_iter": [300]},
               "rf": {"n_estimators": [10], "max_depth": [5, None]},
               "mlp": {"hidden_layer_sizes": [[16]], "alpha": [1e-4, 1e-3], "max_iter": [30]}}




# ── data ────────────────────────────────────────────────────────────────────
@dataclass
class Windows:
    """Window features (and labels, if annotated) per recording. Computed once, reused everywhere."""
    X: dict[str, pd.DataFrame]
    y: dict[str, np.ndarray]           # empty arrays for unannotated recordings
    times: dict[str, np.ndarray]       # timestamp of each window's last uplink
    raw: dict[str, pd.DataFrame]
    columns: list[str]
    classes: list[str]                 # chronological phase order
    window_size: int
    stride: int
    feature_set: str = "all"

    @property
    def recordings(self) -> list[str]:
        return list(self.X)

    def select(self, feature_set: str) -> "Windows":
        """Same windows restricted to the signals of a feature set (see FEATURE_SETS)."""
        signals = FEATURE_SETS[feature_set]
        cols = [c for c in self.columns if feature_signal(c) in signals]
        return Windows({g: x[cols] for g, x in self.X.items()}, self.y, self.times, self.raw, cols,
                       self.classes, self.window_size, self.stride, feature_set)

    def without_raw(self) -> "Windows":
        """Copy without the raw rows: what worker processes need (less to transfer)."""
        return replace(self, raw={})

    def stack(self, exclude=None):
        names = [g for g in self.recordings if g != exclude]
        X = pd.concat([self.X[g] for g in names], ignore_index=True)
        y = np.concatenate([self.y[g] for g in names])
        groups = np.concatenate([np.full(len(self.y[g]), g) for g in names])
        return X, y, groups

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"recording": g, "rows": len(self.raw[g]), "windows": len(self.X[g]),
                              "from": self.times[g][0], "to": self.times[g][-1],
                              **{c: int((self.y[g] == c).sum()) for c in self.classes}} for g in self.recordings])


def feature_signal(column: str) -> str | None:
    """'lse01_temperature_diff_mean' -> 'lse01_temperature_diff'; 'timestamp_interval' -> 'timestamp'."""
    if column.startswith("timestamp_") and column[len("timestamp_"):] in TIMING_STATS:
        return "timestamp"
    signal, _, stat = column.rpartition("_")
    return signal if stat in SIGNAL_STATS else None


def check_air_temperature(windows: Windows, log=print) -> None:
    """Feature sets with temperature differences need the air temperature in the converted data."""
    diff_cols = [c for c in windows.columns if c.endswith("_temperature_diff_mean")]
    if not diff_cols:
        return
    X = pd.concat(list(windows.X.values()), ignore_index=True)
    coverage = float(X[diff_cols].notna().any(axis=1).mean())
    if coverage == 0:
        raise ValueError("No air temperature in the converted data. Re-run notebook 01 (conversion) and check "
                         "the reported air temperature column.")
    if coverage < 0.5:
        log(f"WARNING: only {coverage:.0%} of the windows have an air temperature.")


def augment_temperature(X: pd.DataFrame, y: np.ndarray, settings: dict | None = None, seed: int = RANDOM_STATE):
    """Training-only augmentation of the temperature features. Returns (X, y) with augmented copies added.

    Per window it applies T' = mean + m·b·(T − mean) + a to every temperature signal: offset a (absolute
    temperatures only, the same for both probes; differences keep their level), swing scale b and
    mirror m = ±1. The window statistics have closed forms for this, so it works on the features directly.
    """
    s = {**AUGMENTATION, **(settings or {})}
    signals = sorted({feature_signal(c) for c in X.columns
                      if feature_signal(c) and "temperature" in feature_signal(c)})
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


def _window_recording(path: str, window_size: int, stride: int, annotated: bool):
    g = recording_name(path)
    df = load_recording(path)
    if annotated:
        if "label" not in df.columns or df["label"].isna().all():
            raise ValueError(f"{path} has no labels")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
    raw = df[FEATURE_ORDER]
    X = WindowFeatures(window_size=window_size, stride=stride).transform(raw)
    y = window_labels(df["label"], window_size, stride) if annotated else np.array([], dtype=object)
    times = df["timestamp"].to_numpy()[np.arange(len(X)) * stride + window_size - 1]
    return g, raw, X, y, times


def load_windows(data_dir: str = CONVERTED_DIR, pattern: str = "series_1_*.csv", window_size: int = 10,
                 stride: int = 2, annotated: bool = True, classes=None, n_jobs: int = N_JOBS, log=print) -> Windows:
    """Window features of all recordings matching pattern. annotated=False for series 2."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files match {pattern} in {data_dir}")
    X, y, times, raw = {}, {}, {}, {}
    jobs = Parallel(n_jobs=min(effective_n_jobs(n_jobs), len(paths)), return_as="generator", **NO_MEMMAP)(
        delayed(_window_recording)(p, window_size, stride, annotated) for p in paths)
    for g, raw[g], X[g], y[g], times[g] in jobs:
        log(f"{g}: {len(raw[g]):,} rows → {len(X[g]):,} windows")
    # one fixed feature schema for all recordings (timing features last)
    columns = sorted(set().union(*(set(w.columns) for w in X.values())), key=lambda c: (c.startswith("timestamp"), c))
    X = {g: w.reindex(columns=columns) for g, w in X.items()}
    if classes is None:
        classes = in_phase_order(np.concatenate(list(y.values()))) if annotated else []
    return Windows(X, y, times, raw, columns, list(classes), window_size, stride)


# ── models ──────────────────────────────────────────────────────────────────
def make_classifier(key: str, params: dict, n_jobs: int = 1):
    p = {k: (tuple(v) if k == "hidden_layer_sizes" else v) for k, v in params.items()}
    if key == "logreg":
        # One-vs-rest: liblinear only supports it, and it keeps all solvers comparable.
        return OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE, **p))
    if key == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **p)
    if key == "mlp":
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    raise KeyError(key)


def build_steps(key: str, params: dict, n_jobs: int = 1) -> Pipeline:
    """The trainable part: everything after windowing. n_jobs: threads of the random forest."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", make_classifier(key, params, n_jobs))])


def build_pipeline(fitted: Pipeline, windows: Windows, key: str, name: str) -> ModelPipeline:
    """Fitted steps + window features = a pipeline that takes raw uplink rows."""
    wf = WindowFeatures(window_size=windows.window_size, stride=windows.stride, output_columns=windows.columns)
    return ModelPipeline([("windows", wf)] + list(fitted.steps), name=name,
                         model_type=MODEL_NAMES[key], feature_order=list(FEATURE_ORDER))


def training_params(key: str, params: dict, limit_size: bool, quick: bool) -> dict:
    p = dict(params)
    if limit_size:
        for k, cap in SIZE_LIMITS.get(key, {}).items():
            p[k] = cap if p.get(k) is None else min(p[k], cap)
    if quick:
        p.update(QUICK_PARAMS.get(key, {}))
    return p


# ── hyperparameter search ───────────────────────────────────────────────────
def search_hyperparameters(windows: Windows, keys=("logreg", "rf", "mlp"), grids=None, scoring="balanced_accuracy",
                           n_jobs=-1, verbose=1, log=print) -> dict:
    """Successive-halving grid search with leave-one-sausage-out CV on pre-computed windows.

    Uses the feature set the windows were selected with. Returns {key: {"params", "score", "results"}}.
    """
    grids = grids or PARAM_GRIDS
    X, y, groups = windows.stack()
    prefix = {"logreg": "classifier__estimator__"}
    out = {}
    for key in keys:
        log(f"\n=== {MODEL_NAMES[key]} ===  (feature set: {windows.feature_set})")
        grid = grids[key] if isinstance(grids[key], list) else [grids[key]]
        pre = prefix.get(key, "classifier__")
        grid = [{pre + k: [tuple(x) if isinstance(x, list) else x for x in v] for k, v in g.items()} for g in grid]
        search = HalvingGridSearchCV(build_steps(key, {}), grid, factor=4, resource="n_samples",
                                     min_resources=max(len(X) // (len(windows.recordings) * 10), 50),
                                     cv=LeaveOneGroupOut(), scoring=scoring, random_state=RANDOM_STATE,
                                     n_jobs=n_jobs, verbose=verbose, error_score="raise")
        t = time.perf_counter()
        with parallel_config(**NO_MEMMAP):
            search.fit(X, y, groups=groups)
        best = {k[len(pre):]: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()}
        res = pd.DataFrame(search.cv_results_)
        res = res[res["iter"] == res["iter"].max()].sort_values("rank_test_score")
        table = pd.DataFrame([{k[len(pre):]: v for k, v in p.items()} for p in res["params"]])
        table["balanced_accuracy"] = res["mean_test_score"].round(3).to_numpy()
        table["std"] = res["std_test_score"].round(3).to_numpy()
        out[key] = {"params": best, "score": float(search.best_score_), "results": table}
        log(f"best {scoring} = {search.best_score_:.3f} in {(time.perf_counter() - t) / 60:.1f} min: {best}")
    return out


def save_hyperparameters(result: dict, windows: Windows, out_dir: str = MODELS_DIR, scoring="balanced_accuracy") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, HYPERPARAMETERS_FILE)
    with open(path, "w") as fh:
        json.dump({"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "scoring": scoring,
                   "feature_set": windows.feature_set, "window_size": windows.window_size, "stride": windows.stride,
                   "models": {k: {"params": v["params"], "score": v["score"]} for k, v in result.items()}}, fh, indent=2)
    return path


def load_hyperparameters(out_dir: str = MODELS_DIR) -> tuple[dict, str, str | None]:
    """(params per model, source description, feature set they were searched with).

    Falls back to DEFAULT_PARAMS for every model without a search result.
    """
    path = os.path.join(out_dir, HYPERPARAMETERS_FILE)
    if not os.path.exists(path):
        return dict(DEFAULT_PARAMS), "defaults (no hyperparameters.json yet)", None
    with open(path) as fh:
        saved = json.load(fh)
    params = {k: saved["models"].get(k, {}).get("params", DEFAULT_PARAMS[k]) for k in DEFAULT_PARAMS}
    return params, f"{HYPERPARAMETERS_FILE} from {saved['created_at'][:10]}", saved.get("feature_set")


# ── label-free check on the unannotated run ─────────────────────────────────
def predict_progression(steps: Pipeline, unannotated: Windows, classes: list[str]) -> dict[str, pd.Series]:
    """Predicted phase index (0, 1, 2 in chronological order) over time, smoothed, per recording."""
    order = {c: i for i, c in enumerate(classes)}
    out = {}
    for g in unannotated.recordings:
        if not len(unannotated.X[g]):
            continue
        idx = pd.Series([order[p] for p in steps.predict(unannotated.X[g])],
                        index=pd.DatetimeIndex(unannotated.times[g]), dtype=float)
        smooth = idx.rolling(PROGRESSION_SMOOTHING).median()
        # the first days: incomplete smoothing window and probes settling in; not scored
        warmup = min(pd.Timedelta(PROGRESSION_WARMUP), (smooth.index[-1] - smooth.index[0]) / 10)
        out[g] = smooth[smooth.index >= smooth.index[0] + warmup]
    return out


def _backward_share(s: pd.Series) -> float:
    if not len(s):
        return float("nan")
    sustained = s.rolling(PROGRESSION_SUSTAINED).median()
    return float((sustained < sustained.cummax()).mean())


def progression_scores(progression: dict[str, pd.Series], classes: list[str], until: str | None = PROGRESSION_UNTIL) -> dict:
    """How plausible the predicted phases are on a run without labels.

    spearman:       rank correlation of time and predicted phase. +1 = phases only advance,
                    near 0 = no order, negative = phases run backwards (e.g. following a falling temperature).
    backward_share: share of time the phase, smoothed over PROGRESSION_SUSTAINED, is below a phase it reached earlier
                    (short flickers between neighbouring phases do not count).
    phase_share:    share of time predicted in each phase.
    until:          only score the time before this date (see PROGRESSION_UNTIL); None = everything.
    """
    per = {}
    for g, s in progression.items():
        if until is not None and len(s):
            s = s[s.index < pd.Timestamp(until, tz=s.index.tz)]
        ranks = pd.Series(np.arange(len(s)), dtype=float)
        rho = ranks.corr(s.reset_index(drop=True), method="spearman") if s.nunique() > 1 else np.nan
        per[g] = {"spearman": None if np.isnan(rho) else float(rho),
                  "backward_share": _backward_share(s),
                  "phase_share": {c: float((s.round() == i).mean()) for i, c in enumerate(classes)}}
    rhos = [v["spearman"] for v in per.values() if v["spearman"] is not None]
    return {"spearman_mean": float(np.mean(rhos)) if rhos else None,
            "backward_share_mean": float(np.mean([v["backward_share"] for v in per.values()])) if per else None,
            "recordings": per}


# ── evaluation ──────────────────────────────────────────────────────────────
def _agg(folds, name):
    v = np.array([f[name] for f in folds])
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def _fit_task(windows: Windows, key: str, params: dict, held: str | None, augmentation: dict | None,
              unannotated: Windows | None = None) -> dict:
    """One fit: without `held` (a CV fold) or on all sausages (held=None). Runs in a worker process."""
    X_tr, y_tr, _ = windows.stack(exclude=held)
    if augmentation is not None:
        X_tr, y_tr = augment_temperature(X_tr, y_tr, augmentation)
    steps = build_steps(key, params)
    t = time.perf_counter()
    steps.fit(X_tr, y_tr)
    out = {"held": held, "steps": steps, "fit_seconds": time.perf_counter() - t}
    if held is not None:
        t = time.perf_counter()
        out["y_pred"] = steps.predict(windows.X[held])
        out["pred_seconds"] = time.perf_counter() - t
    if unannotated is not None:
        out["progression"] = predict_progression(steps, unannotated, windows.classes)
    return out


def _run_tasks(tasks: list[tuple], n_jobs: int):
    """Run (_fit_task args) in worker processes; yields (task, result) in task order as results arrive."""
    n = min(effective_n_jobs(n_jobs), len(tasks))
    results = Parallel(n_jobs=n, return_as="generator", **NO_MEMMAP)(delayed(_fit_task)(*t) for t in tasks) if n > 1 \
        else (_fit_task(*t) for t in tasks)
    yield from zip(tasks, results)


def _fold_scores(windows: Windows, results: list[dict], log) -> dict:
    classes = windows.classes
    folds, cm, held_out_pred = [], np.zeros((len(classes), len(classes)), dtype=int), {}
    for r in results:
        held, y_pred, y_true = r["held"], r["y_pred"], windows.y[r["held"]]
        held_out_pred[held] = y_pred
        cm += confusion_matrix(y_true, y_pred, labels=classes)
        folds.append({"held_out": held, "accuracy": accuracy_score(y_true, y_pred),
                      "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                      "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                      "n_windows": int(len(y_true)), "fit_seconds": r["fit_seconds"],
                      "batch_ms_per_window": r["pred_seconds"] * 1000 / max(len(y_true), 1)})
    return {"folds": folds, "confusion_matrix": cm, "held_out_pred": held_out_pred,
            "accuracy": _agg(folds, "accuracy"), "balanced_accuracy": _agg(folds, "balanced_accuracy"),
            "macro_f1": _agg(folds, "macro_f1"), "batch_ms_per_window": _agg(folds, "batch_ms_per_window")["mean"]}


def _log_fold(r: dict, log) -> None:
    if r["held"] is None:
        log(f"  all sausages: fitted ({r['fit_seconds']:.1f} s)")
    else:
        log(f"  without {r['held']}: balanced acc {r['balanced_acc']:.3f} ({r['fit_seconds']:.1f} s)")


def cross_validate(windows: Windows, key: str, params: dict, augmentation: dict | None = None,
                   on_fold=None, n_jobs: int = N_JOBS, log=print) -> dict:
    """Leave-one-sausage-out: fit on all but one annotated sausage, score on that one. Folds run in parallel.

    augmentation: settings for augment_temperature, applied to the training part only.
    on_fold(held_out, fitted_steps) is called after each fold (used to export fold models).
    """
    w = windows.without_raw()
    tasks = [(w, key, params, held, augmentation) for held in w.recordings]
    results = []
    for _, r in _run_tasks(tasks, n_jobs):
        r["balanced_acc"] = balanced_accuracy_score(w.y[r["held"]], r["y_pred"])
        _log_fold(r, log)
        if on_fold:
            on_fold(r["held"], r["steps"])
        results.append(r)
    return _fold_scores(w, results, log)


def compare_feature_sets(windows: Windows, unannotated: Windows | None, variants=None, keys=("logreg", "rf", "mlp"),
                         params: dict | None = None, limit_size: bool = True, quick: bool = False,
                         n_jobs: int = N_JOBS, log=print):
    """Held-out score on the annotated run vs. phase progression on the unannotated run, per variant.

    variants: feature set names; append "+augmentation" to train that set with augment_temperature.
    A model that learned the season scores well on series 1 but runs backwards on series 2.
    All fits (variants × models × folds, plus one fit on all sausages each) share one pool of worker processes.
    Returns (table, progressions) where progressions[(variant, key)] = {recording: Series}.
    """
    params = params or load_hyperparameters()[0]
    variants = variants or list(FEATURE_SETS)
    tasks, meta = [], []
    for variant in variants:
        fs, _, aug = variant.partition("+")
        augmentation = AUGMENTATION if aug == "augmentation" else None
        w = windows.select(fs).without_raw()
        check_air_temperature(w, log)
        u = unannotated.select(fs).without_raw() if unannotated is not None else None
        for key in keys:
            p = training_params(key, params[key], limit_size, quick)
            for held in w.recordings + ([None] if u is not None else []):
                tasks.append((w, key, p, held, augmentation, u if held is None else None))
                meta.append((variant, key))
    log(f"{len(tasks)} fits ({len(variants)} variants × {len(keys)} models × folds) in parallel …")
    done = {}
    for (task, r), (variant, key) in zip(_run_tasks(tasks, n_jobs), meta):
        w = task[0]
        if r["held"] is not None:
            r["balanced_acc"] = balanced_accuracy_score(w.y[r["held"]], r["y_pred"])
        done.setdefault((variant, key), []).append(r)
        if len(done[(variant, key)]) == len(w.recordings) + (unannotated is not None):
            folds = [x for x in done[(variant, key)] if x["held"] is not None]
            accs = [x["balanced_acc"] for x in folds]
            log(f"{variant} · {MODEL_NAMES[key]}: balanced acc {np.mean(accs):.3f} "
                f"(folds {', '.join(f'{a:.2f}' for a in accs)})")
    rows, progressions = [], {}
    for variant in variants:
        for key in keys:
            rs = done[(variant, key)]
            w = windows.select(variant.partition("+")[0])
            cv = _fold_scores(w, [r for r in rs if r["held"] is not None], log)
            row = {"variant": variant, "model": MODEL_NAMES[key], "features": len(w.columns),
                   "series 1: balanced acc": round(cv["balanced_accuracy"]["mean"], 3),
                   "± std": round(cv["balanced_accuracy"]["std"], 3)}
            final = [r for r in rs if r["held"] is None]
            if final:
                progressions[(variant, key)] = final[0]["progression"]
                sc = progression_scores(progressions[(variant, key)], w.classes)
                row["series 2: phase order ρ"] = None if sc["spearman_mean"] is None else round(sc["spearman_mean"], 3)
                row["series 2: backwards share"] = round(sc["backward_share_mean"], 3)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["variant", "model"]), progressions


# ── training and export ─────────────────────────────────────────────────────
def _latency_ms(pipe: ModelPipeline, raw: pd.DataFrame, ws: int, repeats: int = 30) -> float:
    """Median wall time for ONE prediction from raw rows, windowing included."""
    sample = raw.iloc[:ws]
    pipe.predict_proba(sample)  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        pipe.predict_proba(sample)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def train_and_evaluate(windows: Windows, out_dir: str = MODELS_DIR, feature_set: str = DEFAULT_FEATURE_SET,
                       unannotated: Windows | None = None, params: dict | None = None, augmentation: dict | None = None,
                       keys=("logreg", "rf", "mlp"), limit_size: bool = True, fold_models: bool = True,
                       quick: bool = False, n_jobs: int = N_JOBS, log=print) -> dict:
    """Evaluate, export fold and final models, write manifest.json and metrics.json. Returns the metrics.

    windows: all annotated windows (any feature set; feature_set selects the signals).
    unannotated: windows of the unannotated run for the phase-progression check (optional).
    augmentation: settings for augment_temperature (training data only), None = off.
    """
    params_source, searched_with = "given", None
    if params is None:
        params, params_source, searched_with = load_hyperparameters(out_dir)
    log(f"Feature set: {feature_set} ({', '.join(FEATURE_SETS[feature_set])})")
    log(f"Hyperparameters: {params_source}")
    if searched_with and searched_with != feature_set:
        log(f"WARNING: hyperparameters were searched with feature set '{searched_with}'. Re-run notebook 02.")
    windows = windows.select(feature_set)
    check_air_temperature(windows, log)
    if augmentation is not None:
        log(f"Augmentation of temperature features: {augmentation}")
    if unannotated is not None:
        unannotated = unannotated.select(feature_set)
    classes, recs = windows.classes, windows.recordings
    metrics, manifest_models, held_out_pred = {}, [], {}

    for key in keys:
        p = training_params(key, params[key], limit_size, quick)
        log(f"\n=== {MODEL_NAMES[key]} ===  {p}")
        fold_entries, results, light = {}, [], windows.without_raw()
        u = unannotated.without_raw() if unannotated is not None else None
        tasks = [(light, key, p, held, augmentation) for held in recs] + [(light, key, p, None, augmentation, u)]
        for _, r in _run_tasks(tasks, n_jobs):
            if r["held"] is not None:
                r["balanced_acc"] = balanced_accuracy_score(windows.y[r["held"]], r["y_pred"])
                if fold_models:
                    d = os.path.join(out_dir, "folds", r["held"])
                    name = f"{key}_without_{r['held']}"
                    build_pipeline(r["steps"], windows, key, name).store(d)
                    fold_entries[r["held"]] = {"dir": os.path.relpath(d, out_dir), "name": name}
            _log_fold(r, log)
            results.append(r)
        cv = _fold_scores(windows, [r for r in results if r["held"] is not None], log)
        held_out_pred[key] = cv["held_out_pred"]
        final_run = next(r for r in results if r["held"] is None)
        steps, fit_all = final_run["steps"], final_run["fit_seconds"]
        final = build_pipeline(steps, windows, key, f"{key}_final")
        size = os.path.getsize(final.store(os.path.join(out_dir, "final")))
        latency = _latency_ms(final, windows.raw[recs[0]], windows.window_size)
        metrics[key] = {"name": MODEL_NAMES[key], "params": p, "folds": cv["folds"],
                        "accuracy": cv["accuracy"], "balanced_accuracy": cv["balanced_accuracy"],
                        "macro_f1": cv["macro_f1"], "confusion_matrix": cv["confusion_matrix"].tolist(),
                        "single_prediction_ms": latency, "batch_ms_per_window": cv["batch_ms_per_window"],
                        "fit_seconds_all_data": fit_all, "model_file_bytes": size}
        line = (f"  mean balanced acc {cv['balanced_accuracy']['mean']:.3f} · "
                f"{latency:.1f} ms per prediction · {size / 1e6:.1f} MB")
        if unannotated is not None:
            sc = progression_scores(final_run["progression"], classes)
            metrics[key]["unannotated_progression"] = sc
            if sc["spearman_mean"] is not None:
                line += f" · series 2 phase order ρ {sc['spearman_mean']:+.2f}"
        log(line)
        manifest_models.append({"key": key, "name": MODEL_NAMES[key],
                                "final": {"dir": "final", "name": f"{key}_final"}, "folds": fold_entries})

    keys = list(metrics)
    agreement = {f"{a}|{b}": float(np.mean(np.concatenate([held_out_pred[a][g] for g in recs])
                                           == np.concatenate([held_out_pred[b][g] for g in recs])))
                 for i, a in enumerate(keys) for b in keys[i + 1:]}
    # training feature ranges for the demonstrator's out-of-range warning
    X_all, y_all, _ = windows.stack()
    ranges = {}
    for c in windows.columns:
        v = X_all[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            ranges[c] = [float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))]

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {"created_at": now, "window_size": windows.window_size, "stride": windows.stride,
                "feature_set": feature_set, "signals": FEATURE_SETS[feature_set], "augmentation": augmentation,
                "feature_order": FEATURE_ORDER, "output_columns": windows.columns, "classes": classes,
                "labeled_groups": recs, "models": manifest_models, "feature_ranges": ranges,
                "quick": quick, "hyperparameters": params_source, "size_limited": limit_size,
                "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__}}
    result = {"created_at": now, "classes": classes, "models": metrics, "agreement_on_held_out": agreement,
              "quick": quick, "feature_set": feature_set, "signals": FEATURE_SETS[feature_set],
              "augmentation": augmentation,
              "unannotated_recordings": unannotated.recordings if unannotated is not None else [],
              "dataset": {"labeled_groups": recs, "n_windows": int(len(y_all)),
                          "class_counts": {c: int((y_all == c).sum()) for c in classes},
                          "window_size": windows.window_size, "stride": windows.stride}}
    os.makedirs(out_dir, exist_ok=True)
    for fname, obj in (("manifest.json", manifest), ("metrics.json", result)):
        with open(os.path.join(out_dir, fname), "w") as fh:
            json.dump(obj, fh, indent=2)
    log(f"\nWrote models, manifest.json and metrics.json to {out_dir}")
    return result


def metrics_table(metrics: dict) -> pd.DataFrame:
    """One row per model, for display in the notebook."""
    rows = []
    for m in metrics["models"].values():
        row = {"model": m["name"],
               "balanced acc": f"{m['balanced_accuracy']['mean']:.3f} ± {m['balanced_accuracy']['std']:.3f}",
               "macro-F1": f"{m['macro_f1']['mean']:.3f} ± {m['macro_f1']['std']:.3f}",
               "accuracy": f"{m['accuracy']['mean']:.3f} ± {m['accuracy']['std']:.3f}",
               "ms / prediction": round(m["single_prediction_ms"], 2),
               "size (MB)": round(m["model_file_bytes"] / 1e6, 2),
               "fit all data (s)": round(m["fit_seconds_all_data"], 1)}
        prog = m.get("unannotated_progression")
        if prog:
            row["series 2: phase order ρ"] = None if prog["spearman_mean"] is None else round(prog["spearman_mean"], 3)
            row["series 2: backwards share"] = round(prog["backward_share_mean"], 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train and evaluate the demonstrator models (same as notebook 03).")
    ap.add_argument("--data-dir", default=CONVERTED_DIR)
    ap.add_argument("--out", default=MODELS_DIR)
    ap.add_argument("--feature-set", default=DEFAULT_FEATURE_SET, choices=list(FEATURE_SETS))
    ap.add_argument("--augment", action="store_true", help="augment temperature features (training data only)")
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no-size-limit", action="store_true", help="export the searched Random Forest without caps")
    ap.add_argument("--no-fold-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny models, for smoke tests only")
    ap.add_argument("--jobs", type=int, default=N_JOBS, help="worker processes (-1 = all cores)")
    args = ap.parse_args(argv)
    windows = load_windows(args.data_dir, window_size=args.window_size, stride=args.stride, n_jobs=args.jobs)
    unannotated = load_windows(args.data_dir, "series_2_*.csv", args.window_size, args.stride,
                               annotated=False, classes=windows.classes, n_jobs=args.jobs)
    train_and_evaluate(windows, args.out, args.feature_set, unannotated,
                       augmentation=AUGMENTATION if args.augment else None, limit_size=not args.no_size_limit,
                       fold_models=not args.no_fold_models, quick=args.quick, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
