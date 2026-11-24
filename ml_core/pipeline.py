"""sklearn Pipeline with a fixed input column order and stored metadata."""
from __future__ import annotations

import datetime
import json
import os
from typing import Sequence

import joblib
from sklearn.pipeline import Pipeline


class ModelPipeline(Pipeline):
    """Raw uplink rows in, phase prediction out (window features → imputer → scaler → classifier).

    ``feature_order`` selects and orders the raw input columns, so callers can pass
    frames with extra columns. ``store``/``load`` write ``<name>_model.joblib`` plus a
    readable ``<name>_metadata.json`` next to it.
    """

    def __init__(self, steps: Sequence[tuple], name: str, model_type: str, feature_order: list,
                 created_at: datetime.datetime | None = None):
        super().__init__(steps)
        self.name = name
        self.model_type = model_type
        self.feature_order = feature_order
        self.created_at = created_at if created_at is not None else datetime.datetime.now()

    def metadata(self) -> dict:
        return {"name": self.name, "model_type": self.model_type,
                "feature_order": self.feature_order, "created_at": self.created_at.isoformat()}

    def store(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}_model.joblib")
        joblib.dump(self, path)
        with open(os.path.join(directory, f"{self.name}_metadata.json"), "w") as fh:
            json.dump(self.metadata(), fh, indent=2)
        return path

    @staticmethod
    def load(directory: str, name: str) -> "ModelPipeline":
        return joblib.load(os.path.join(directory, f"{name}_model.joblib"))

    def fit(self, X, y=None, **fit_params):
        # Windowing turns n rows into fewer windows, but sklearn passes y through
        # unchanged, so a plain fit cannot work. Fit the steps after windowing on
        # pre-computed windows instead (see ml_core.training.build_pipeline).
        raise NotImplementedError("fit the post-windowing steps on windows; see ml_core.training")

    def predict(self, X, **params):
        return super().predict(X[self.feature_order], **params)

    def predict_proba(self, X, **params):
        return super().predict_proba(X[self.feature_order], **params)
