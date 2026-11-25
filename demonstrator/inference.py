"""CPU inference for the three model families, with a compatibility check."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from ml_core import ModelPipeline, WindowFeatures
from ml_core.data import FEATURE_ORDER

log = logging.getLogger(__name__)
# share of in-range-checked features allowed outside the training range
OOD_THRESHOLD = 0.25


def single_threaded(pipe):
    """One window per call: thread start-up costs more than it saves (and drains battery)."""
    clf = pipe.steps[-1][1]
    for est in [clf] + list(getattr(clf, "estimators_", []) if hasattr(clf, "n_jobs") else []):
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    return pipe


def latest_values(rows: pd.DataFrame) -> dict:
    """Most recent reading per measurement in the window (for displays). Air = probe − (probe − air)."""
    out = {}
    for c in ("lse01_moisture", "lse01_conductivity", "lse01_temperature", "ltcfs_temperature"):
        v = rows[c].dropna() if c in rows else []
        if len(v):
            out[c] = float(v.iloc[-1])
    for probe in ("ltcfs", "lse01"):
        both = rows[[f"{probe}_temperature", f"{probe}_temperature_diff"]].dropna() if f"{probe}_temperature_diff" in rows else []
        if len(both):
            out["ambient_temperature"] = float(both.iloc[-1, 0] - both.iloc[-1, 1])
            break
    return out


class ModelRegistry:
    """Loads final and per-fold pipelines lazily from data/models/manifest.json."""

    def __init__(self, models_dir: str):
        """Load manifest.json, metrics.json and the final models from `models_dir`."""
        self.models_dir = models_dir
        self.manifest: dict | None = None
        self.metrics: dict | None = None
        self.error: str | None = None
        self._cache: dict[tuple, ModelPipeline] = {}
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
        """(Re)load the manifest, metrics and final models. Errors are kept in self.error (shown in the UI)."""
        self._cache.clear()
        self.manifest, self.error = None, None
        mpath = os.path.join(self.models_dir, "manifest.json")
        try:
            with open(mpath) as fh:
                self.manifest = json.load(fh)
        except FileNotFoundError:
            self.error = f"No models found ({mpath}). Run notebooks/03_train_and_evaluate.ipynb first."
            return
        try:
            with open(os.path.join(self.models_dir, "metrics.json")) as fh:
                self.metrics = json.load(fh)
        except FileNotFoundError:
            self.metrics = None
        try:
            for m in self.manifest["models"]:
                self._load(m["key"], None)  # fail early on the final models
        except Exception as e:  # noqa: BLE001
            self.error = f"Model loading failed: {e!r}"
            log.exception("model loading failed")
        w = self.manifest
        self._wf = WindowFeatures(window_size=w["window_size"], stride=w["stride"],
                                   output_columns=w["output_columns"], context=w.get("context", ()))

    @property
    def ok(self) -> bool:
        """True if models are loaded and usable."""
        return self.manifest is not None and self.error is None

    @property
    def window_size(self) -> int:
        """Uplinks per window the models were trained with."""
        return self.manifest["window_size"] if self.manifest else 10

    @property
    def stride(self) -> int:
        """Step between windows used in training (the replay predicts at the same step)."""
        return self.manifest["stride"] if self.manifest else 2

    @property
    def history(self) -> pd.Timedelta:
        """History the models need before the current window (longest context horizon; 0 without context)."""
        return self._wf.history if self.manifest else pd.Timedelta(0)

    def history_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The rows a prediction needs: the last window plus the context history before it (FEATURE_ORDER columns)."""
        rows = rows[FEATURE_ORDER]
        if self.history <= pd.Timedelta(0) or not len(rows):
            return rows.tail(self.window_size).reset_index(drop=True)
        ts = pd.to_datetime(rows["timestamp"], utc=True)
        keep = (ts > ts.iloc[-1] - self.history).to_numpy().copy()
        keep[-self.window_size:] = True
        return rows[keep].reset_index(drop=True)

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Window features of the last window of `rows` (one row, the models' input before imputing)."""
        return self._wf.transform_last(self.history_rows(rows))

    @property
    def keys(self) -> list[str]:
        """Model keys in manifest order (logreg, rf, mlp)."""
        return [m["key"] for m in self.manifest["models"]] if self.manifest else []

    def info(self) -> dict:
        """Model summary for /api/state, with a warning if scikit-learn differs from training."""
        if not self.manifest:
            return {"ok": False, "error": self.error, "models": []}
        trained_with = self.manifest.get("versions", {}).get("scikit-learn")
        import sklearn
        warning = None
        if trained_with and trained_with != sklearn.__version__:
            warning = (f"models trained with scikit-learn {trained_with}, running {sklearn.__version__}: "
                       "retrain or install the same version")
        return {"ok": self.ok, "error": self.error, "version_warning": warning,
                "models": [{"key": m["key"], "name": m["name"],
                            "has_fold_models": bool(m.get("folds"))} for m in self.manifest["models"]],
                "classes": self.manifest["classes"],
                "window_size": self.window_size, "stride": self.stride,
                "feature_set": self.manifest.get("feature_set", "all"), "signals": self.manifest.get("signals"),
                "created_at": self.manifest.get("created_at"), "quick": self.manifest.get("quick")}

    def _entry(self, key):
        """The manifest entry of one model."""
        for m in self.manifest["models"]:
            if m["key"] == key:
                return m
        raise KeyError(key)

    def _load(self, key: str, held_out: str | None) -> ModelPipeline:
        """Load (and cache) the final model (held_out=None) or the fold model without `held_out`."""
        ck = (key, held_out)
        with self._lock:
            if ck not in self._cache:
                m = self._entry(key)
                e = m["folds"][held_out] if held_out else m["final"]
                pipe = ModelPipeline.load(os.path.join(self.models_dir, e["dir"]), e["name"])
                single_threaded(pipe)
                pipe.after_windows = Pipeline(pipe.steps[1:])   # imputer → scaler → classifier, for ready features
                self._cache[ck] = pipe
            return self._cache[ck]

    def model_for(self, key: str, group: str | None):
        """Fold model that never saw `group` if available, else the final model."""
        m = self._entry(key)
        if group and group in m.get("folds", {}):
            return self._load(key, group), False
        seen = bool(group) and group in self.manifest.get("labeled_groups", [])
        return self._load(key, None), seen

    def ood_fraction(self, rows: pd.DataFrame) -> float:
        """Share of window features outside the training range (0.5–99.5 percentile ± 5 %).
        Above OOD_THRESHOLD the dashboard warns that the input is unlike the training data.
        """
        return self._ood(self.features(rows).iloc[-1])

    def _ood(self, feats: pd.Series) -> float:
        """ood_fraction for ready features."""
        ranges = self.manifest.get("feature_ranges", {})
        checked = outside = 0
        for c, (lo, hi) in ranges.items():
            v = feats.get(c)
            if v is None or not np.isfinite(v):
                continue
            checked += 1
            margin = 0.05 * (hi - lo)
            outside += int(v < lo - margin or v > hi + margin)
        return outside / checked if checked else 1.0

    def predict(self, rows: pd.DataFrame, group: str | None = None) -> dict:
        """Predict on the most recent window of raw rows (FEATURE_ORDER columns; with context models, pass the
        history too: rows of the last `history` before the window, see history_rows).

        The window features are computed once and fed to every model; each model's "ms" is the feature time plus
        its own time, i.e. what one prediction costs on its own.
        """
        ws = self.window_size
        hist = self.history_rows(rows)
        if len(hist) < ws:
            return {"ready": False, "reason": f"need {ws} uplinks, have {len(hist)}"}
        t = time.perf_counter()
        feats = self._wf.transform_last(hist)
        feat_ms = (time.perf_counter() - t) * 1000
        rows = hist.tail(ws).reset_index(drop=True)
        out = {"ready": True, "models": {}, "any_model_saw_data": False}
        for key in self.keys:
            pipe, seen = self.model_for(key, group)
            t = time.perf_counter()
            proba = pipe.after_windows.predict_proba(feats)[-1]
            ms = feat_ms + (time.perf_counter() - t) * 1000
            classes = [str(c) for c in pipe.classes_]
            i = int(np.argmax(proba))
            out["models"][key] = {"label": classes[i], "confidence": float(proba[i]),
                                  "proba": {c: float(p) for c, p in zip(classes, proba)},
                                  "ms": ms, "model_saw_this_recording": seen}
            out["any_model_saw_data"] |= seen
        labels = [m["label"] for m in out["models"].values()]
        out["agreement_count"] = max(labels.count(x) for x in labels)  # size of the largest group of equal answers
        out["agreement"] = ("all" if out["agreement_count"] == len(labels)
                            else "none" if out["agreement_count"] == 1 else "partial")
        out["values"] = latest_values(rows)
        out["ood_fraction"] = self._ood(feats.iloc[-1])
        out["compatible"] = out["ood_fraction"] <= OOD_THRESHOLD
        return out
