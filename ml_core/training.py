"""Hyperparameter search, leave-one-sausage-out evaluation and model export.

Used by notebooks/02_search_hyperparameters.ipynb and 03_train_and_evaluate.ipynb.
Headless alternative for the training step:  python -m ml_core.training

Output in data/models/ (what the demonstrator loads):
    hyperparameters.json   search result (written by the search, read by training)
    manifest.json          model registry, window settings, feature schema and ranges, library versions
    metrics.json           per-fold scores, confusion matrices, latency, size, model agreement
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
from dataclasses import dataclass

import numpy as np
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
# Tiny settings for smoke tests (notebook QUICK flag, test suite).
QUICK_PARAMS = {"rf": {"n_estimators": 20}, "mlp": {"max_iter": 30, "batch_size": 128}}
QUICK_GRIDS = {"logreg": {"solver": ["lbfgs"], "l1_ratio": [0], "C": [0.1, 10], "max_iter": [300]},
               "rf": {"n_estimators": [10], "max_depth": [5, None]},
               "mlp": {"hidden_layer_sizes": [[16]], "alpha": [1e-4, 1e-3], "max_iter": [30]}}


# ── data ────────────────────────────────────────────────────────────────────
@dataclass
class Windows:
    """Window features and labels per labeled recording (computed once, reused everywhere)."""
    X: dict[str, pd.DataFrame]
    y: dict[str, np.ndarray]
    raw: dict[str, pd.DataFrame]
    columns: list[str]
    classes: list[str]
    window_size: int
    stride: int

    @property
    def recordings(self) -> list[str]:
        return list(self.X)

    def stack(self, only=None, exclude=None):
        names = [g for g in self.recordings if (only is None or g in only) and g != exclude]
        X = pd.concat([self.X[g] for g in names], ignore_index=True)
        y = np.concatenate([self.y[g] for g in names])
        groups = np.concatenate([np.full(len(self.y[g]), g) for g in names])
        return X, y, groups

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{"recording": g, "rows": len(self.raw[g]), "windows": len(self.y[g]),
                              **{c: int((self.y[g] == c).sum()) for c in self.classes}} for g in self.recordings])


def load_windows(data_dir: str = CONVERTED_DIR, pattern: str = "series_1_*.csv",
                 window_size: int = 10, stride: int = 2, log=print) -> Windows:
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files match {pattern} in {data_dir}")
    wf = WindowFeatures(window_size=window_size, stride=stride)
    X, y, raw = {}, {}, {}
    for p in paths:
        g = recording_name(p)
        df = load_recording(p)
        if "label" not in df.columns or df["label"].isna().all():
            raise ValueError(f"{p} has no labels")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        raw[g] = df[FEATURE_ORDER]
        X[g] = wf.transform(raw[g])
        y[g] = window_labels(df["label"], window_size, stride)
        log(f"{g}: {len(df):,} rows → {len(X[g]):,} windows")
    # one fixed feature schema for all recordings (timing features last)
    columns = sorted(set().union(*(set(w.columns) for w in X.values())), key=lambda c: (c.startswith("timestamp"), c))
    X = {g: w.reindex(columns=columns) for g, w in X.items()}
    classes = in_phase_order(np.concatenate(list(y.values())))
    return Windows(X, y, raw, columns, classes, window_size, stride)


# ── models ──────────────────────────────────────────────────────────────────
def make_classifier(key: str, params: dict):
    p = {k: (tuple(v) if k == "hidden_layer_sizes" else v) for k, v in params.items()}
    if key == "logreg":
        # One-vs-rest: liblinear only supports it, and it keeps all solvers comparable.
        return OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE, **p))
    if key == "rf":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
    if key == "mlp":
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    raise KeyError(key)


def build_steps(key: str, params: dict) -> Pipeline:
    """The trainable part: everything after windowing."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", make_classifier(key, params))])


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

    Returns {key: {"params", "score", "results"}} where results is a DataFrame of all candidates.
    """
    grids = grids or PARAM_GRIDS
    X, y, groups = windows.stack()
    prefix = {"logreg": "classifier__estimator__"}
    out = {}
    for key in keys:
        log(f"\n=== {MODEL_NAMES[key]} ===")
        grid = grids[key] if isinstance(grids[key], list) else [grids[key]]
        pre = prefix.get(key, "classifier__")
        grid = [{pre + k: [tuple(x) if isinstance(x, list) else x for x in v] for k, v in g.items()} for g in grid]
        search = HalvingGridSearchCV(build_steps(key, {}), grid, factor=4, resource="n_samples",
                                     min_resources=max(len(X) // (len(windows.recordings) * 10), 50),
                                     cv=LeaveOneGroupOut(), scoring=scoring, random_state=RANDOM_STATE,
                                     n_jobs=n_jobs, verbose=verbose, error_score="raise")
        t = time.perf_counter()
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
                   "window_size": windows.window_size, "stride": windows.stride,
                   "models": {k: {"params": v["params"], "score": v["score"]} for k, v in result.items()}}, fh, indent=2)
    return path


def load_hyperparameters(out_dir: str = MODELS_DIR) -> tuple[dict, str]:
    """(params per model, source description). Falls back to DEFAULT_PARAMS per missing model."""
    path = os.path.join(out_dir, HYPERPARAMETERS_FILE)
    if not os.path.exists(path):
        return dict(DEFAULT_PARAMS), "defaults (no hyperparameters.json yet)"
    with open(path) as fh:
        saved = json.load(fh)
    params = {k: saved["models"].get(k, {}).get("params", DEFAULT_PARAMS[k]) for k in DEFAULT_PARAMS}
    return params, f"{HYPERPARAMETERS_FILE} from {saved['created_at'][:10]}"


# ── evaluation and export ───────────────────────────────────────────────────
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


def _agg(folds, name):
    v = np.array([f[name] for f in folds])
    return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


def train_and_evaluate(windows: Windows, out_dir: str = MODELS_DIR, params: dict | None = None,
                       keys=("logreg", "rf", "mlp"), limit_size: bool = True, fold_models: bool = True,
                       quick: bool = False, log=print) -> dict:
    """Leave-one-sausage-out evaluation, final models, manifest.json and metrics.json. Returns the metrics."""
    params_source = "given"
    if params is None:
        params, params_source = load_hyperparameters(out_dir)
    log(f"Hyperparameters: {params_source}")
    classes, recs = windows.classes, windows.recordings
    metrics, manifest_models, held_out_pred = {}, [], {}

    for key in keys:
        p = training_params(key, params[key], limit_size, quick)
        log(f"\n=== {MODEL_NAMES[key]} ===  {p}")
        folds, cm = [], np.zeros((len(classes), len(classes)), dtype=int)
        fold_entries, held_out_pred[key] = {}, {}
        for held in recs:
            X_tr, y_tr, _ = windows.stack(exclude=held)
            steps = build_steps(key, p)
            t = time.perf_counter()
            steps.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            y_pred = steps.predict(windows.X[held])
            pred_s = time.perf_counter() - t
            y_true = windows.y[held]
            held_out_pred[key][held] = y_pred
            cm += confusion_matrix(y_true, y_pred, labels=classes)
            f = {"held_out": held, "accuracy": accuracy_score(y_true, y_pred),
                 "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                 "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                 "n_windows": int(len(y_true)), "fit_seconds": fit_s,
                 "batch_ms_per_window": pred_s * 1000 / max(len(y_true), 1)}
            folds.append(f)
            log(f"  without {held}: balanced acc {f['balanced_accuracy']:.3f}, macro-F1 {f['macro_f1']:.3f} ({fit_s:.1f} s)")
            if fold_models:
                d = os.path.join(out_dir, "folds", held)
                name = f"{key}_without_{held}"
                build_pipeline(steps, windows, key, name).store(d)
                fold_entries[held] = {"dir": os.path.relpath(d, out_dir), "name": name}

        X_all, y_all, _ = windows.stack()
        steps = build_steps(key, p)
        t = time.perf_counter()
        steps.fit(X_all, y_all)
        fit_all = time.perf_counter() - t
        final = build_pipeline(steps, windows, key, f"{key}_final")
        size = os.path.getsize(final.store(os.path.join(out_dir, "final")))
        clf = final.steps[-1][1]
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1  # one window per call in the demonstrator; threads only add overhead
        latency = _latency_ms(final, windows.raw[recs[0]], windows.window_size)
        metrics[key] = {"name": MODEL_NAMES[key], "params": p, "folds": folds,
                        "accuracy": _agg(folds, "accuracy"), "balanced_accuracy": _agg(folds, "balanced_accuracy"),
                        "macro_f1": _agg(folds, "macro_f1"), "confusion_matrix": cm.tolist(),
                        "single_prediction_ms": latency, "batch_ms_per_window": _agg(folds, "batch_ms_per_window")["mean"],
                        "fit_seconds_all_data": fit_all, "model_file_bytes": size}
        log(f"  mean balanced acc {metrics[key]['balanced_accuracy']['mean']:.3f} · "
            f"{latency:.1f} ms per prediction · {size / 1e6:.1f} MB")
        manifest_models.append({"key": key, "name": MODEL_NAMES[key],
                                "final": {"dir": "final", "name": f"{key}_final"}, "folds": fold_entries})

    # how often two models agree on held-out windows
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
                "feature_order": FEATURE_ORDER, "output_columns": windows.columns, "classes": classes,
                "labeled_groups": recs, "models": manifest_models, "feature_ranges": ranges,
                "quick": quick, "hyperparameters": params_source, "size_limited": limit_size,
                "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__}}
    result = {"created_at": now, "classes": classes, "models": metrics, "agreement_on_held_out": agreement,
              "quick": quick, "dataset": {"labeled_groups": recs, "n_windows": int(len(y_all)),
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
    for key, m in metrics["models"].items():
        rows.append({"model": m["name"],
                     "balanced acc": f"{m['balanced_accuracy']['mean']:.3f} ± {m['balanced_accuracy']['std']:.3f}",
                     "macro-F1": f"{m['macro_f1']['mean']:.3f} ± {m['macro_f1']['std']:.3f}",
                     "accuracy": f"{m['accuracy']['mean']:.3f} ± {m['accuracy']['std']:.3f}",
                     "ms / prediction": round(m["single_prediction_ms"], 2),
                     "size (MB)": round(m["model_file_bytes"] / 1e6, 2),
                     "fit all data (s)": round(m["fit_seconds_all_data"], 1)})
    return pd.DataFrame(rows).set_index("model")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train and evaluate the demonstrator models (same as notebook 03).")
    ap.add_argument("--data-dir", default=CONVERTED_DIR)
    ap.add_argument("--out", default=MODELS_DIR)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--no-size-limit", action="store_true", help="export the searched Random Forest without caps")
    ap.add_argument("--no-fold-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny models, for smoke tests only")
    args = ap.parse_args(argv)
    windows = load_windows(args.data_dir, window_size=args.window_size, stride=args.stride)
    train_and_evaluate(windows, args.out, limit_size=not args.no_size_limit,
                       fold_models=not args.no_fold_models, quick=args.quick)


if __name__ == "__main__":
    main()
