"""Explain a single prediction: raw uplinks -> features -> per-model reasoning.

Two kinds of explanation, on purpose:

* "Which signals mattered" uses ONE model-agnostic test for all three models
  (replace a sensor signal's features by their typical training value and see
  how much the predicted class's probability drops). Same method for every
  model, so the bars are comparable side by side.
* "How this model computes" is model-specific and exact for the model at hand:
  the weighted sum of the logistic regression, the votes and split
  contributions of the random forest (Saabas method), and the neuron
  activations of the MLP. These show the three different principles.

Occlusion and Saabas contributions are approximations of "importance";
correlated signals share credit unpredictably. The UI says so.

Only keys and numbers are returned; all labels and texts are translated in
the frontend (demonstrator/static/i18n.js).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import FEATURE_ORDER

SIGNALS = ["lse01_conductivity", "lse01_moisture", "lse01_temperature", "ltcfs_temperature", "timestamp"]
UNITS = {"lse01_moisture": "%", "lse01_conductivity": "µS/cm", "lse01_temperature": "°C",
         "ltcfs_temperature": "°C", "timestamp": "s"}


def describe_feature(name: str) -> tuple[str, str]:
    """feature column -> (signal key, statistic key), e.g. lse01_moisture_mean -> (lse01_moisture, mean)."""
    for sig in sorted(SIGNALS, key=len, reverse=True):
        if name.startswith(sig + "_"):
            return sig, name[len(sig) + 1:]
    return name, ""


def _steps(pipe):
    s = dict(pipe.steps)
    return s["windows"], s["imputer"], s["scaler"], pipe.steps[-1][1]


def _proba(clf, x):
    return clf.predict_proba(x)[0]


def _logreg(clf, xs, names, k):
    est = clf.estimators_[k] if hasattr(clf, "estimators_") else clf
    coef = est.coef_[0] if est.coef_.shape[0] == 1 else est.coef_[k]
    intercept = float(np.ravel(est.intercept_)[0 if est.coef_.shape[0] == 1 else k])
    contrib = coef * xs[0]
    order = np.argsort(-np.abs(contrib))
    top = [{"feature": names[i], "weight": float(coef[i]), "scaled_value": float(xs[0, i]), "contribution": float(contrib[i])}
           for i in order[:8]]
    rest = float(contrib[order[8:]].sum())
    score = float(contrib.sum() + intercept)
    return {"kind": "weighted_sum", "top": top, "rest": rest, "intercept": intercept,
            "score": score, "n_features": int(len(contrib))}


def _forest(clf, xs, names, k):
    n_classes = len(clf.classes_)
    votes = np.zeros(n_classes, dtype=int)
    contrib = np.zeros(xs.shape[1])
    tree_votes = []
    for est in clf.estimators_:
        p = est.predict_proba(xs)[0]
        v = int(np.argmax(p))
        votes[v] += 1
        tree_votes.append(v)
        t = est.tree_
        path = est.decision_path(xs).indices
        val = t.value[:, 0, :]
        val = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-12)
        for parent, child in zip(path[:-1], path[1:]):
            contrib[t.feature[parent]] += val[child, k] - val[parent, k]
    contrib /= len(clf.estimators_)
    order = np.argsort(-np.abs(contrib))
    top = [{"feature": names[i], "contribution": float(contrib[i])} for i in order[:8] if contrib[i] != 0]
    depth = [int(e.get_depth()) for e in clf.estimators_]
    return {"kind": "votes", "classes": [str(c) for c in clf.classes_],
            "votes": votes.tolist(), "tree_votes": tree_votes, "n_trees": len(clf.estimators_),
            "max_depth": max(depth), "top": top}


def _mlp(clf, xs, k):
    a = xs[0]
    layers = [{"name": "input", "size": int(a.size), "act": _norm(np.abs(a))}]
    acts = [a]
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        z = a @ W + b
        last = i == len(clf.coefs_) - 1
        if not last:
            a = np.maximum(z, 0) if clf.activation == "relu" else np.tanh(z)
            layers.append({"name": f"hidden {i + 1}", "size": int(a.size), "act": _norm(a),
                           "active": int((a > 1e-9).sum())})
            acts.append(a)
    prob = _proba(clf, xs)
    layers.append({"name": "output", "size": int(prob.size), "act": [float(p) for p in prob]})
    W_last = clf.coefs_[-1][:, k] * acts[-1]
    strongest = [int(i) for i in np.argsort(-W_last)[:5] if W_last[i] > 0]
    n_params = int(sum(W.size + b.size for W, b in zip(clf.coefs_, clf.intercepts_)))
    return {"kind": "network", "layers": layers, "strongest_last_hidden": strongest,
            "classes": [str(c) for c in clf.classes_],
            "n_params": n_params, "activation": clf.activation}


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _norm(v) -> list[float]:
    v = np.asarray(v, dtype=float)
    m = float(np.max(np.abs(v))) if v.size else 0.0
    return [round(float(x) / m, 3) if m > 0 else 0.0 for x in v]


def explain_model(pipe, key: str, rows: pd.DataFrame) -> dict:
    wm, imputer, scaler, clf = _steps(pipe)
    feats = wm.transform(rows[FEATURE_ORDER]).tail(1)
    names = list(imputer.get_feature_names_out())
    xi = imputer.transform(feats)
    xs = scaler.transform(xi)
    base = scaler.transform(imputer.statistics_[[list(imputer.feature_names_in_).index(n) for n in names]]
                            .reshape(1, -1))
    classes = [str(c) for c in clf.classes_]
    p = _proba(clf, xs)
    k = int(np.argmax(p))

    # same occlusion test for every model, per sensor signal
    groups = {}
    for i, n in enumerate(names):
        groups.setdefault(describe_feature(n)[0], []).append(i)
    occlusion = []
    for sig, idx in groups.items():
        x2 = xs.copy()
        x2[0, idx] = base[0, idx]
        p2 = _proba(clf, x2)
        # log-odds change stays informative when a model is ~100 % sure
        occlusion.append({"signal": sig, "n_features": len(idx),
                          "p_before": float(p[k]), "p_after": float(p2[k]),
                          "evidence": float(_logit(p[k]) - _logit(p2[k])),
                          "would_predict": classes[int(np.argmax(p2))]})
    occlusion.sort(key=lambda o: -abs(o["evidence"]))

    if key == "logreg":
        detail = _logreg(clf, xs, names, k)
    elif key == "rf":
        detail = _forest(clf, xs, names, k)
    elif key == "mlp":
        detail = _mlp(clf, xs, k)
    else:
        detail = {"kind": "none"}
    return {"label": classes[k], "proba": {c: float(v) for c, v in zip(classes, p)},
            "occlusion": occlusion, "detail": detail, "n_features_used": len(names)}


def explain(registry, rows: pd.DataFrame, group: str | None) -> dict:
    ws = registry.window_size
    rows = rows[FEATURE_ORDER].tail(ws).reset_index(drop=True)
    if len(rows) < ws:
        return {"ready": False, "reason": f"need {ws} uplinks, have {len(rows)}"}
    any_pipe, _ = registry.model_for(registry.keys[0], group)
    feats = any_pipe.steps[0][1].transform(rows).iloc[-1]
    ranges = registry.manifest.get("feature_ranges", {})
    features = []
    for name, v in feats.items():
        sig, stat_key = describe_feature(name)
        lo, hi = ranges.get(name, [None, None])
        pos = None
        if lo is not None and np.isfinite(v) and hi > lo:
            pos = float((v - lo) / (hi - lo))
        features.append({"feature": name, "signal": sig, "stat_key": stat_key,
                         "unit": UNITS.get(sig, ""), "value": None if not np.isfinite(v) else float(v),
                         "range": [lo, hi], "position": pos})
    uplinks = []
    for _, r in rows.iterrows():
        vals = {c: float(r[c]) for c in FEATURE_ORDER[2:] if pd.notna(r[c])}
        uplinks.append({"time": pd.Timestamp(r["timestamp"]).isoformat(), "sensor": r["sensor"], "values": vals})
    t0, t1 = pd.Timestamp(rows["timestamp"].iloc[0]), pd.Timestamp(rows["timestamp"].iloc[-1])
    models = {}
    for key in registry.keys:
        pipe, seen = registry.model_for(key, group)
        models[key] = explain_model(pipe, key, rows) | {"model_saw_this_recording": seen}
    return {"ready": True, "uplinks": uplinks, "window_seconds": (t1 - t0).total_seconds(),
            "features": features, "models": models}
