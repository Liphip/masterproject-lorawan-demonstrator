"""Train, evaluate and export the three demonstrator models.

Replaces the (currently non-runnable) final cells 12-13 of
01_create_pipelines.ipynb. Hyperparameter search stays in the notebook; this
script takes its winners (or demo-sized variants) and produces everything the
demonstrator needs:

    data/models/manifest.json        model registry + window config + feature ranges
    data/models/metrics.json         leave-one-sausage-out evaluation
    data/models/final/*.joblib       models fitted on all labeled sausages
    data/models/folds/<group>/*      models fitted WITHOUT <group> (honest replay)

Usage (from the repository root):
    python training/train_models.py                       # demo-sized RF
    python training/train_models.py --rf-preset hpo       # exact notebook winners
    python training/train_models.py --no-fold-models      # smaller output
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from ml_app import CustomPipeline, FeatureExtractor, WindowingMethod  # noqa: E402
from ml_app.data import (FEATURE_ORDER, group_name, load_converted_csv,  # noqa: E402
                         window_labels)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Skipping features without any observed values")

RANDOM_STATE = 42

# Winners of the HalvingGridSearchCV run in 01_create_pipelines.ipynb (cell 10).
HPO_PARAMS = {
    # penalty="l2" is liblinear's default; the explicit argument is deprecated in sklearn>=1.8
    "logreg": dict(C=100, solver="liblinear", class_weight=None, max_iter=1000),
    "rf": dict(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=4,
               max_features="sqrt", bootstrap=True, class_weight="balanced_subsample",
               criterion="gini"),
    "mlp": dict(hidden_layer_sizes=(64, 32, 16), activation="relu", alpha=1e-4,
                learning_rate_init=1e-3, batch_size=32, max_iter=500,
                early_stopping=True, validation_fraction=0.15),
}
# RF with unlimited depth on ~160k windows grows to hundreds of MB per model.
# The demo preset bounds depth/trees so it stays small and fast on a laptop CPU.
RF_DEMO = dict(HPO_PARAMS["rf"], n_estimators=100, max_depth=10)

MODEL_NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "mlp": "MLP (neural net)"}


def make_classifier(key: str, rf_preset: str, quick: bool):
    if key == "logreg":
        # liblinear was one-vs-rest implicitly in older sklearn; sklearn>=1.8 requires
        # the explicit wrapper for >2 classes. Same model as the notebook's winner.
        return OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE, **HPO_PARAMS["logreg"]))
    if key == "rf":
        p = dict(HPO_PARAMS["rf"] if rf_preset == "hpo" else RF_DEMO)
        if quick:
            p["n_estimators"] = 20
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **p)
    if key == "mlp":
        p = dict(HPO_PARAMS["mlp"])
        if quick:
            p.update(max_iter=30, batch_size=128)
        return MLPClassifier(random_state=RANDOM_STATE, **p)
    raise KeyError(key)


def inner_pipeline(clf) -> Pipeline:
    # step name "inputer" kept identical to the notebook pipelines
    return Pipeline([("inputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", clf)])


def wrap(inner: Pipeline, wm: WindowingMethod, key: str, version: str) -> CustomPipeline:
    return CustomPipeline([("windowing", wm)] + list(inner.steps), version=version,
                          model_type=MODEL_NAMES[key], feature_order=list(FEATURE_ORDER))


def dir_size(path: str) -> int:
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "*")) if os.path.isfile(p))


def latency_ms(pipe: CustomPipeline, raw: pd.DataFrame, ws: int, repeats: int = 30) -> float:
    """Median wall time for ONE prediction from raw rows (windowing included)."""
    sample = raw.iloc[:ws]
    pipe.predict_proba(sample)  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        pipe.predict_proba(sample)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "training", "converted"))
    ap.add_argument("--labeled-glob", default="series_1_*.csv")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "models"))
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--models", default="logreg,rf,mlp")
    ap.add_argument("--rf-preset", choices=["demo", "hpo"], default="demo")
    ap.add_argument("--no-fold-models", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny models, for smoke tests only")
    args = ap.parse_args(argv)

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    ws, st = args.window_size, args.stride
    paths = sorted(glob.glob(os.path.join(args.data_dir, args.labeled_glob)))
    if not paths:
        sys.exit(f"No files match {args.labeled_glob} in {args.data_dir}")

    # ── 1. window every labeled sausage once ────────────────────────────────
    print(f"Windowing {len(paths)} recordings (window={ws} rows, stride={st}) ...")
    t0 = time.time()
    raw, windows, labels = {}, {}, {}
    wm_free = WindowingMethod(method=FeatureExtractor(), window_size=ws, stride=st)
    for p in paths:
        g = group_name(p)
        df = load_converted_csv(p)
        if "label" not in df.columns or df["label"].isna().all():
            sys.exit(f"{p} has no labels")
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        raw[g] = df[FEATURE_ORDER]
        windows[g] = wm_free.transform(raw[g])
        labels[g] = window_labels(df["label"], ws, st)
        print(f"  {g}: {len(df):,} rows -> {len(windows[g]):,} windows")
    output_columns = sorted(set().union(*[set(w.columns) for w in windows.values()]),
                            key=lambda c: (c.startswith("timestamp"), c))
    for g in windows:
        windows[g] = windows[g].reindex(columns=output_columns)
    wm = WindowingMethod(method=FeatureExtractor(), window_size=ws, stride=st,
                         output_columns=output_columns)
    groups = list(windows)
    classes = sorted(set(np.concatenate(list(labels.values()))))
    print(f"  {len(output_columns)} features, classes={classes}  ({time.time() - t0:.1f}s)")

    os.makedirs(os.path.join(args.out, "final"), exist_ok=True)
    save_folds = not args.no_fold_models

    # ── 2. leave-one-sausage-out evaluation ─────────────────────────────────
    metrics = {}
    manifest_models = []
    oof = {k: {} for k in keys}
    for key in keys:
        print(f"\n== {MODEL_NAMES[key]} ==")
        folds, cm_total = [], np.zeros((len(classes), len(classes)), dtype=int)
        fold_entries = {}
        for held in groups:
            X_tr = pd.concat([windows[g] for g in groups if g != held], ignore_index=True)
            y_tr = np.concatenate([labels[g] for g in groups if g != held])
            inner = inner_pipeline(make_classifier(key, args.rf_preset, args.quick))
            t = time.perf_counter()
            inner.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t
            t = time.perf_counter()
            y_pred = inner.predict(windows[held])
            pred_s = time.perf_counter() - t
            y_true = labels[held]
            oof[key][held] = y_pred
            cm_total += confusion_matrix(y_true, y_pred, labels=classes)
            f = dict(held_out=held,
                     accuracy=accuracy_score(y_true, y_pred),
                     balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                     macro_f1=f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
                     n_windows=int(len(y_true)), fit_seconds=fit_s,
                     batch_ms_per_window=pred_s * 1000 / max(len(y_true), 1))
            folds.append(f)
            print(f"  held out {held}: acc={f['accuracy']:.3f} bal_acc={f['balanced_accuracy']:.3f} "
                  f"macroF1={f['macro_f1']:.3f} fit={fit_s:.1f}s")
            if save_folds:
                d = os.path.join(args.out, "folds", held)
                os.makedirs(d, exist_ok=True)
                version = f"{key}_without_{held}"
                wrap(inner, wm, key, version).store(d)
                fold_entries[held] = {"dir": os.path.relpath(d, args.out), "version": version}

        # ── 3. final model on all labeled data ──────────────────────────────
        X_all = pd.concat([windows[g] for g in groups], ignore_index=True)
        y_all = np.concatenate([labels[g] for g in groups])
        inner = inner_pipeline(make_classifier(key, args.rf_preset, args.quick))
        t = time.perf_counter()
        inner.fit(X_all, y_all)
        fit_all = time.perf_counter() - t
        version = f"{key}_final"
        final_pipe = wrap(inner, wm, key, version)
        final_pipe.store(os.path.join(args.out, "final"))
        size = os.path.getsize(os.path.join(args.out, "final", f"{version}_model.joblib"))
        clf = final_pipe.steps[-1][1]
        if hasattr(clf, "n_jobs"):
            clf.n_jobs = 1  # inference is single-window; matches demonstrator/inference.py
        lat = latency_ms(final_pipe, raw[groups[0]], ws)

        def agg(name):
            v = np.array([f[name] for f in folds])
            return {"mean": float(v.mean()), "std": float(v.std(ddof=0)), "min": float(v.min()), "max": float(v.max())}

        metrics[key] = {
            "name": MODEL_NAMES[key],
            "params": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in inner.named_steps["classifier"].get_params().items()
                       if k.replace("estimator__", "") in (HPO_PARAMS[key] if key != "rf" else RF_DEMO)},
            "folds": folds,
            "accuracy": agg("accuracy"), "balanced_accuracy": agg("balanced_accuracy"),
            "macro_f1": agg("macro_f1"),
            "confusion_matrix": cm_total.tolist(),
            "single_prediction_ms": lat,
            "batch_ms_per_window": agg("batch_ms_per_window")["mean"],
            "fit_seconds_all_data": fit_all,
            "model_file_bytes": size,
        }
        print(f"  mean bal_acc={metrics[key]['balanced_accuracy']['mean']:.3f}  "
              f"latency={lat:.2f} ms/prediction  size={size / 1e6:.1f} MB")
        manifest_models.append({"key": key, "name": MODEL_NAMES[key],
                                "final": {"dir": "final", "version": version},
                                "folds": fold_entries})

    # ── 4. agreement between models on held-out windows ─────────────────────
    agreement = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ya = np.concatenate([oof[a][g] for g in groups])
            yb = np.concatenate([oof[b][g] for g in groups])
            agreement[f"{a}|{b}"] = float((ya == yb).mean())

    # ── 5. feature ranges for the live/replay compatibility check ───────────
    X_all = pd.concat([windows[g] for g in groups], ignore_index=True)
    ranges = {}
    for c in output_columns:
        v = X_all[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            ranges[c] = [float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))]

    y_all = np.concatenate([labels[g] for g in groups])
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    dataset = {
        "labeled_groups": groups,
        "n_windows": int(len(y_all)),
        "class_counts": {c: int((y_all == c).sum()) for c in classes},
        "window_size": ws, "stride": st,
        "note": ("All labeled sausages come from ONE curing run (series 1). Leave-one-"
                 "sausage-out separates sausages, not runs; scores are likely optimistic "
                 "for a new run."),
    }
    manifest = {"created_at": now, "window_size": ws, "stride": st,
                "feature_order": FEATURE_ORDER, "output_columns": output_columns,
                "classes": classes, "labeled_groups": groups, "models": manifest_models, "feature_ranges": ranges,
                "quick": bool(args.quick), "rf_preset": args.rf_preset,
                "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__}}
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump({"created_at": now, "classes": classes, "dataset": dataset,
                   "models": metrics, "agreement_on_held_out": agreement,
                   "quick": bool(args.quick)}, fh, indent=2)
    print(f"\nWrote {args.out}/manifest.json and metrics.json")


if __name__ == "__main__":
    main()
