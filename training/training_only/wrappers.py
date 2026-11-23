from sklearn.base import BaseEstimator, ClassifierMixin, clone as sklearn_clone
from sklearn.preprocessing import LabelEncoder
import copy
import numpy as np
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as SklearnPipeline
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
from ml_app import WindowingMethod, FeatureExtractor



class WindowedEstimator(BaseEstimator):
    """Meta-estimator that applies windowing per-group and delegates to an inner sklearn Pipeline.

    This wrapper is training-only: it expands time-series groups into windowed
    feature rows and fits an inner pipeline on those windows. The trained inner
    pipeline is what should be persisted for production.
    """

    def __init__(self, pipeline: Any, window_size: int = 10, stride: int = 2, method: Any = None):
        self.original_pipeline = pipeline
        self.window_size = window_size
        self.stride = stride
        # default feature extractor
        self.method = method if method is not None else FeatureExtractor()

        # build an inner sklearn Pipeline without any 'windowing' step
        steps = []
        for name, step in getattr(pipeline, 'steps', pipeline):
            if name == 'windowing':
                continue
            steps.append((name, step))
        self.pipeline = SklearnPipeline(steps)

    def _create_windows_for_group(self, group_X: pd.DataFrame, group_y: pd.Series):
        wm = WindowingMethod(method=self.method, window_size=self.window_size, stride=self.stride)
        n_rows = len(group_X)
        starts = list(range(0, n_rows - self.window_size + 1, self.stride)) if n_rows >= self.window_size else []
        if len(starts) == 0:
            return pd.DataFrame(), []

        windows = wm.transform(group_X.reset_index(drop=True))
        labels = []
        for start in starts:
            window_labels = group_y.iloc[start:start + self.window_size]
            if len(window_labels) == 0:
                labels.append(None)
            else:
                try:
                    label = window_labels.mode().iloc[0]
                except Exception:
                    label = window_labels.iloc[-1]
                labels.append(label)

        return windows.reset_index(drop=True), labels

    def fit(self, X: pd.DataFrame, y: pd.Series = None, **fit_params):
        if '__groups__' in X.columns:
            groups_arr = np.array(X['__groups__'])
            X = X.drop(columns=['__groups__'])
        else:
            groups_arr = np.array(['__all__'] * len(X))
        X_df = X.reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)

        all_windows = []
        all_labels = []
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            group_X = X_df.loc[mask].reset_index(drop=True)
            group_y = y_ser.loc[mask].reset_index(drop=True)
            wins, labels = self._create_windows_for_group(group_X, group_y)
            if not wins.empty:
                all_windows.append(wins)
                all_labels.extend(labels)

        if len(all_windows) == 0:
            raise ValueError('No windows created. Check window_size/stride and input data.')

        Xw = pd.concat(all_windows, ignore_index=True)
        yw = pd.Series(all_labels)

        self.feature_names_in_ = Xw.columns.tolist()
        self.label_encoder_ = LabelEncoder()
        yw_enc = pd.Series(self.label_encoder_.fit_transform(yw))
        self.pipeline.fit(Xw, yw_enc)
        return self

    def predict(self, X: pd.DataFrame, groups=None):
        if '__groups__' in X.columns:
            groups_arr = np.array(X['__groups__'])
            X = X.drop(columns=['__groups__'])
        else:
            groups_arr = np.array(['__all__'] * len(X))
        X_df = X.reset_index(drop=True)

        all_windows = []
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            group_X = X_df.loc[mask].reset_index(drop=True)
            wins, _ = self._create_windows_for_group(group_X, pd.Series([None]*len(group_X)))
            if not wins.empty:
                all_windows.append(wins)

        if len(all_windows) == 0:
            return np.array([])

        Xw = pd.concat(all_windows, ignore_index=True)
        Xw = Xw.reindex(columns=self.feature_names_in_)
        return self.label_encoder_.inverse_transform(self.pipeline.predict(Xw))

    def predict_proba(self, X: pd.DataFrame, groups=None):
        if '__groups__' in X.columns:
            groups_arr = np.array(X['__groups__'])
            X = X.drop(columns=['__groups__'])
        else:
            groups_arr = np.array(['__all__'] * len(X))
        X_df = X.reset_index(drop=True)

        all_windows = []
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            group_X = X_df.loc[mask].reset_index(drop=True)
            wins, _ = self._create_windows_for_group(group_X, pd.Series([None]*len(group_X)))
            if not wins.empty:
                all_windows.append(wins)

        if len(all_windows) == 0:
            return np.array([])

        Xw = pd.concat(all_windows, ignore_index=True)
        Xw = Xw.reindex(columns=self.feature_names_in_)
        probas = self.pipeline.predict_proba(Xw)
        return probas

    def score(self, X, y, **fit_params):
        y_ser = pd.Series(y).reset_index(drop=True)
        if '__groups__' in X.columns:
            groups_arr = np.array(X['__groups__'])
            X = X.drop(columns=['__groups__'])
        else:
            groups_arr = np.array(['__all__'] * len(X))
        X_df = X.reset_index(drop=True)

        all_windows = []
        all_labels = []
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            group_X = X_df.loc[mask].reset_index(drop=True)
            group_y = y_ser.loc[mask].reset_index(drop=True)
            wins, labels = self._create_windows_for_group(group_X, group_y)
            if not wins.empty:
                all_windows.append(wins)
                all_labels.extend(labels)

        if len(all_windows) == 0:
            return 0.0

        Xw = pd.concat(all_windows, ignore_index=True)
        Xw = Xw.reindex(columns=self.feature_names_in_)

        yw = pd.Series(self.label_encoder_.transform(pd.Series(all_labels)))
        return self.pipeline.score(Xw, yw)


class SearchableWindowedEstimator(BaseEstimator, ClassifierMixin):
    """Clone-safe wrapper that delegates to the production WindowedEstimator.

    Behavior:
    - Stores constructor args exactly so sklearn.clone() succeeds.
    - Captures params starting with 'pipeline__' in set_params and stashes them.
    - On fit(), clones/ deepcopies the provided pipeline, applies stashed pipeline params,
      constructs the real production WindowedEstimator with that pipeline copy, then fits.
    - Delegates predict/predict_proba/score to the inner fitted estimator.
    """

    def __init__(self, pipeline: Any, window_size: int = 10, stride: int = 2, method: Any = None):
        # assign as-is (no mutation) so sklearn.clone can inspect constructor args
        self.pipeline = pipeline
        self.window_size = window_size
        self.stride = stride
        self.method = method
        # stash for pipeline-specific params set via 'pipeline__...'
        self.pipeline_params = {}
        self._inner_estimator = None

    def set_params(self, **params):
        pipeline_updates = {}
        other = {}
        for k, v in params.items():
            if k.startswith("pipeline__"):
                pipeline_updates[k[len("pipeline__"):]] = v
            else:
                other[k] = v
        if pipeline_updates:
            # merge updates; will be applied to a fresh pipeline copy at fit time
            self.pipeline_params.update(pipeline_updates)
        if other:
            return super().set_params(**other)
        return self

    def fit(self, X, y=None, **fit_params):

        # Try to clone the pipeline; fallback to deepcopy if cloning not supported.
        try:
            pipeline_copy = sklearn_clone(self.pipeline)
        except Exception:
            pipeline_copy = copy.deepcopy(self.pipeline)

        # Apply stashed pipeline params to the pipeline copy.
        if self.pipeline_params:
            try:
                pipeline_copy.set_params(**self.pipeline_params)
            except Exception:
                # best-effort: attempt to set params individually
                for k, v in self.pipeline_params.items():
                    try:
                        pipeline_copy.set_params(**{k: v})
                    except Exception:
                        pass

        # construct and fit the real WindowedEstimator
        inner = WindowedEstimator(pipeline=pipeline_copy, window_size=self.window_size, stride=self.stride, method=self.method)
        groups = fit_params.get('groups', None)
        if groups is not None:
            inner.fit(X, y, groups=groups)
        else:
            inner.fit(X, y)
        self._inner_estimator = inner
        return self

    def predict(self, X, **kwargs):
        return self._inner_estimator.predict(X)

    def predict_proba(self, X, **kwargs):
        if hasattr(self._inner_estimator, 'predict_proba'):
            return self._inner_estimator.predict_proba(X)
        raise AttributeError('Underlying estimator has no predict_proba')

    def score(self, X, y=None, **kwargs):
        return self._inner_estimator.score(X, y)
