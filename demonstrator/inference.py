"""CPU inference for the three model families, with a compatibility check."""
from __future__ import annotations

import json
import logging
import os
import threading
import time

import numpy as np
import pandas as pd

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


class ModelRegistry:
    """Loads final and per-fold pipelines lazily from data/models/manifest.json."""

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.manifest: dict | None = None
        self.metrics: dict | None = None
        self.error: str | None = None
        self._cache: dict[tuple, ModelPipeline] = {}
        self._lock = threading.Lock()
        self.reload()

    def reload(self):
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
                                   output_columns=w["output_columns"])

    @property
    def ok(self) -> bool:
        return self.manifest is not None and self.error is None

    @property
    def window_size(self) -> int:
        return self.manifest["window_size"] if self.manifest else 10

    @property
    def stride(self) -> int:
        return self.manifest["stride"] if self.manifest else 2

    @property
    def keys(self) -> list[str]:
        return [m["key"] for m in self.manifest["models"]] if self.manifest else []

    def info(self) -> dict:
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
                "created_at": self.manifest.get("created_at"), "quick": self.manifest.get("quick")}

    def _entry(self, key):
        for m in self.manifest["models"]:
            if m["key"] == key:
                return m
        raise KeyError(key)

    def _load(self, key: str, held_out: str | None) -> ModelPipeline:
        ck = (key, held_out)
        with self._lock:
            if ck not in self._cache:
                m = self._entry(key)
                e = m["folds"][held_out] if held_out else m["final"]
                pipe = ModelPipeline.load(os.path.join(self.models_dir, e["dir"]), e["name"])
                single_threaded(pipe)
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
        feats = self._wf.transform(rows).iloc[-1]
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
        """Predict on the most recent window of raw rows (FEATURE_ORDER columns)."""
        ws = self.window_size
        rows = rows[FEATURE_ORDER].tail(ws).reset_index(drop=True)
        if len(rows) < ws:
            return {"ready": False, "reason": f"need {ws} uplinks, have {len(rows)}"}
        out = {"ready": True, "models": {}, "any_model_saw_data": False}
        for key in self.keys:
            pipe, seen = self.model_for(key, group)
            t = time.perf_counter()
            proba = pipe.predict_proba(rows)[-1]
            ms = (time.perf_counter() - t) * 1000
            classes = [str(c) for c in pipe.classes_]
            i = int(np.argmax(proba))
            out["models"][key] = {"label": classes[i], "confidence": float(proba[i]),
                                  "proba": {c: float(p) for c, p in zip(classes, proba)},
                                  "ms": ms, "model_saw_this_recording": seen}
            out["any_model_saw_data"] |= seen
        labels = {m["label"] for m in out["models"].values()}
        out["agreement"] = "all" if len(labels) == 1 else ("none" if len(labels) == len(out["models"]) else "partial")
        out["ood_fraction"] = self.ood_fraction(rows)
        out["compatible"] = out["ood_fraction"] <= OOD_THRESHOLD
        return out
