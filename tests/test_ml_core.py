"""ml_core: window features, conversion, search → training hand-off, notebooks."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

from ml_core import ModelPipeline, WindowFeatures
from ml_core.conversion import classify_device, convert_all
from ml_core.data import ALL_COLUMNS, FEATURE_ORDER, ROOT, add_temperature_difference, load_recording
from ml_core.training import (FEATURE_SETS, QUICK_GRIDS, augment_temperature, compare_feature_sets, feature_signal,
                              load_hyperparameters, load_windows, save_hyperparameters, search_hyperparameters,
                              train_and_evaluate)
from synthetic_data import make_raw_workbooks


def _quiet(*a, **k):
    pass


# ── window features ─────────────────────────────────────────────────────────
def _rows():
    t = pd.date_range("2024-01-01", periods=10, freq="60s", tz="UTC")
    return pd.DataFrame({"timestamp": t,
                         "sensor": ["ws202", "lse01", "ltcfs"] * 3 + ["ws202"],
                         "lse01_conductivity": [np.nan, 100, np.nan, np.nan, 110, np.nan, np.nan, 130, np.nan, np.nan],
                         "lse01_moisture": np.nan, "lse01_temperature": np.nan,
                         "ltcfs_temperature": [np.nan, np.nan, 10, np.nan, np.nan, 11, np.nan, np.nan, 13, np.nan]})


def test_slope_uses_valid_readings_and_any_timestamp_resolution():
    df = _rows()
    out_ns = WindowFeatures(window_size=10, stride=2).transform(df)
    assert out_ns["lse01_conductivity_slope"].iloc[0] == pytest.approx(30 / 6)
    assert out_ns["ltcfs_temperature_slope"].iloc[0] == pytest.approx(3 / 6)
    df_us = df.assign(timestamp=df["timestamp"].astype("datetime64[us, UTC]"))
    out_us = WindowFeatures(window_size=10, stride=2).transform(df_us)
    assert out_us["timestamp_duration"].iloc[0] == pytest.approx(540.0) == out_ns["timestamp_duration"].iloc[0]


def test_output_columns_fixed_schema():
    df = _rows()
    df = df[df["sensor"] != "ltcfs"]  # a sensor missing from the window
    cols = ["lse01_conductivity_mean", "ltcfs_temperature_mean", "timestamp_interval"]
    out = WindowFeatures(window_size=5, stride=1, output_columns=cols).transform(df)
    assert list(out.columns) == cols and out["ltcfs_temperature_mean"].isna().all()


def test_pipeline_fit_explains_itself():
    with pytest.raises(NotImplementedError, match="windows"):
        ModelPipeline([("windows", WindowFeatures())], name="x", model_type="x", feature_order=["timestamp"]).fit(_rows())


# ── conversion ──────────────────────────────────────────────────────────────
def test_classify_device_order_matters():
    assert classify_device("Wurst 1 Feuchtigkeit, Temperatur & Leitwert") == ("lse01", 1)  # also contains "Temperatur"
    assert classify_device("Wurst 3 Temperatur") == ("ltcfs", 3)
    assert classify_device("Milesight Umgebung PIR & Lichtintensität") == ("ws202", None)
    assert classify_device("Wurst 2 PH-Wert") == (None, None)
    assert classify_device("Umgebung Temperatur & Luftfeuchtigkeit") == ("ambient", None)


def test_conversion_of_both_layouts(tmp_path):
    make_raw_workbooks(str(tmp_path / "raw"), n_sausages=2, rows_per_device=12)
    summary = convert_all(str(tmp_path / "raw"), str(tmp_path / "converted"), log=_quiet)
    assert sorted(summary["file"]) == [f"series_{s}_sausage_{k}.csv" for s in (1, 2) for k in (1, 2)]
    # air temperature: different column per export, reported, aligned to every probe uplink
    assert set(summary["air temp source"]) == {"sausage_temperature_sensor_1", "temperature2"}
    assert (summary["air temp coverage"] == 1.0).all()
    s1 = pd.read_csv(tmp_path / "converted" / "series_1_sausage_2.csv")
    s2 = pd.read_csv(tmp_path / "converted" / "series_2_sausage_1.csv")
    assert list(s1.columns) == ALL_COLUMNS + ["label"]
    assert list(s2.columns) == ALL_COLUMNS            # unlabeled run: no label column, but NOT empty
    for df in (s1, s2):
        assert df["sensor"].value_counts().to_dict() == {"lse01": 12, "ltcfs": 12, "ws202": 12}  # pH dropped, ambient no rows
        assert df["ambient_temperature"].eq(14.0).all()
        assert set(df["ws202_light_status"].dropna()) == {0, 1}
        assert df["lse01_moisture"].notna().sum() == 12 and df["ltcfs_temperature"].notna().sum() == 12
        assert pd.to_datetime(df["timestamp"]).is_monotonic_increasing
    assert set(s1["label"]) == {"Umroetung", "Reifung Phase 1", "Reifung Phase 2"}


# ── search → training hand-off ──────────────────────────────────────────────
def test_search_result_is_used_by_training(synthetic_dir, tmp_path):
    windows = load_windows(str(synthetic_dir / "converted"), log=_quiet)
    result = search_hyperparameters(windows, keys=("logreg", "rf"), grids=QUICK_GRIDS, n_jobs=1, verbose=0, log=_quiet)
    assert set(result) == {"logreg", "rf"} and "balanced_accuracy" in result["rf"]["results"]
    assert "C" in result["logreg"]["results"].columns                  # prefixes stripped for display
    save_hyperparameters(result, windows, str(tmp_path))
    params, source, feature_set = load_hyperparameters(str(tmp_path))
    assert source.startswith("hyperparameters.json") and feature_set == "all"
    assert params["rf"] == result["rf"]["params"]
    assert params["mlp"]["hidden_layer_sizes"] == [64, 32, 16]       # not searched → default
    assert load_hyperparameters(str(tmp_path / "empty"))[1].startswith("defaults")


def test_training_outputs(trained):
    m = json.load(open(trained / "models" / "metrics.json"))
    man = json.load(open(trained / "models" / "manifest.json"))
    assert set(m["models"]) == {"logreg", "rf", "mlp"}
    assert m["models"]["rf"]["params"]["max_depth"] == 10            # size limit applied
    assert m["classes"] == ["Umroetung", "Reifung Phase 1", "Reifung Phase 2"]  # chronological
    assert man["feature_set"] == "temperature_difference"                # default: no absolute temperature, no timing
    assert {feature_signal(c) for c in man["output_columns"]} == set(FEATURE_SETS["temperature_difference"])
    assert m["models"]["rf"]["unannotated_progression"]["spearman_mean"] > 0.5
    assert man["versions"]["scikit-learn"] and man["hyperparameters"].startswith("defaults")
    for e in man["models"]:
        assert ModelPipeline.load(os.path.join(trained / "models", e["final"]["dir"]), e["final"]["name"]).steps[0][0] == "windows"


# ── seasonal confound ───────────────────────────────────────────────────────
def test_progression_check_catches_the_season(synthetic_dir):
    """Synthetic data like the real runs: temperature rises over series 1, falls over series 2.
    With temperature the models look perfect on series 1 but run backwards on series 2."""
    w = load_windows(str(synthetic_dir / "converted"), log=_quiet)
    u = load_windows(str(synthetic_dir / "converted"), "series_2_*.csv", annotated=False, classes=w.classes, log=_quiet)
    # exact selection: lse01_temperature must not pick up lse01_temperature_diff
    assert not any("diff" in c for c in w.select("original").columns)
    assert {feature_signal(c) for c in w.select("temperature_difference").columns} == set(FEATURE_SETS["temperature_difference"])
    variants = ["original", "temperature_difference", "temperature_difference+temperature",
                "temperature_difference+calibration+robust"]
    table, prog = compare_feature_sets(w, u, variants, keys=("logreg",), quick=True, log=_quiet)
    rho = table["series 2: phase order ρ"]
    assert rho[("original", "Logistic Regression")] < 0.8 < rho[("temperature_difference", "Logistic Regression")]
    assert rho[("temperature_difference+temperature", "Logistic Regression")] > 0.8
    assert rho[("temperature_difference+calibration+robust", "Logistic Regression")] > 0.8
    assert (table["worst sausage"] <= table["series 1: balanced acc"]).all()
    assert set(prog[("original", "logreg")]) == set(u.recordings)


def test_temperature_difference_uses_recorded_or_given_air_temperature():
    df = pd.DataFrame({"lse01_temperature": [10.0, np.nan], "ltcfs_temperature": [np.nan, 12.0], "ambient_temperature": [8.0, 9.0]})
    add_temperature_difference(df)
    assert df["lse01_temperature_diff"].iloc[0] == 2.0 and df["ltcfs_temperature_diff"].iloc[1] == 3.0
    add_temperature_difference(df, ambient=20.0)                     # demonstrator case: configured air temperature
    assert df["lse01_temperature_diff"].iloc[0] == -10.0 and df["ambient_temperature"].eq(20.0).all()


def test_augmentation_equals_transforming_the_raw_temperatures():
    """Feature-level augmentation must match T' = mean + m·b·(T - mean) + a on the raw rows of one window."""
    rng = np.random.default_rng(3)
    t = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    raw = pd.DataFrame({"timestamp": t, "sensor": ["lse01", "ltcfs"] * 5,
                        "lse01_conductivity": np.nan, "lse01_moisture": np.nan,
                        "lse01_temperature": rng.normal(15, 2, 10), "ltcfs_temperature": rng.normal(15, 2, 10),
                        "ambient_temperature": rng.normal(14, 3, 10)})
    raw.loc[raw.sensor == "ltcfs", "lse01_temperature"] = np.nan
    raw.loc[raw.sensor == "lse01", "ltcfs_temperature"] = np.nan
    add_temperature_difference(raw)
    wf = WindowFeatures(window_size=10, stride=1)
    X = wf.transform(raw[FEATURE_ORDER])
    cols = [c for c in X.columns if "temperature" in c]
    Xa, _ = augment_temperature(X[cols], np.array(["x"]), {"copies": 1}, seed=11)
    # draw the same random numbers as augment_temperature does for one window
    r = np.random.default_rng(11)
    a, b = r.uniform(-10, 10, 1)[0], r.uniform(0.5, 2.0, 1)[0]
    m = -1.0 if r.random(1)[0] < 0.5 else 1.0
    raw2 = raw.copy()
    for col in ("lse01_temperature", "ltcfs_temperature", "lse01_temperature_diff", "ltcfs_temperature_diff"):
        mean = raw2[col].mean()
        raw2[col] = mean + m * b * (raw2[col] - mean) + (0.0 if col.endswith("_diff") else a)
    expected = wf.transform(raw2[FEATURE_ORDER])[cols]
    np.testing.assert_allclose(Xa.iloc[1].to_numpy(float), expected.iloc[0].to_numpy(float), rtol=1e-5, atol=1e-5)


# ── notebooks ───────────────────────────────────────────────────────────────
NOTEBOOKS = os.path.join(ROOT, "notebooks")


def _run_notebook(name: str, overrides: str):
    """Execute a notebook with settings overridden right after its 'parameters' cell."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb = nbformat.read(os.path.join(NOTEBOOKS, name), as_version=4)
    i = next(i for i, c in enumerate(nb.cells) if "parameters" in c.metadata.get("tags", []))
    nb.cells.insert(i + 1, nbformat.v4.new_code_cell(overrides))
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": NOTEBOOKS}}).execute()
    return nb


def test_notebooks_run_end_to_end(tmp_path):
    raw, conv, models = tmp_path / "raw", tmp_path / "converted", tmp_path / "models"
    make_raw_workbooks(str(raw), n_sausages=2, rows_per_device=60)
    _run_notebook("01_convert_raw_data.ipynb", f"RAW_DIR = {str(raw)!r}\nCONVERTED_DIR = {str(conv)!r}")
    assert len(list(conv.glob("*.csv"))) == 4
    common = f"CONVERTED_DIR = {str(conv)!r}\nMODELS_DIR = {str(models)!r}\nQUICK = True"
    _run_notebook("02_search_hyperparameters.ipynb", common + "\nN_JOBS = 1")
    assert (models / "hyperparameters.json").exists()
    _run_notebook("03_train_and_evaluate.ipynb", common)
    assert json.load(open(models / "manifest.json"))["hyperparameters"].startswith("hyperparameters.json")
    assert "unannotated_progression" in json.load(open(models / "metrics.json"))["models"]["rf"]
    _run_notebook("04_check_seasonal_confound.ipynb", f"CONVERTED_DIR = {str(conv)!r}\nQUICK = True")
    _run_notebook("05_compare_windows.ipynb", f"CONVERTED_DIR = {str(conv)!r}\nQUICK = True\n"
                  "CONFIGS = [(10, ()), (10, ('6h',))]\nMODELS = ['logreg']")


def test_parallel_folds_match_sequential(synthetic_dir):
    """Worker processes must not change results: same folds, same predictions as a sequential run."""
    from ml_core.training import cross_validate, training_params
    w = load_windows(str(synthetic_dir / "converted"), n_jobs=2, log=_quiet).select("temperature_difference")
    p = training_params("rf", {"n_estimators": 10, "max_depth": 5}, limit_size=True, quick=False)
    seq = cross_validate(w, "rf", p, n_jobs=1, log=_quiet)
    par = cross_validate(w, "rf", p, n_jobs=2, log=_quiet)
    assert [f["held_out"] for f in seq["folds"]] == [f["held_out"] for f in par["folds"]]
    for g in w.recordings:
        assert (seq["held_out_pred"][g] == par["held_out_pred"][g]).all()


def test_run_notebooks_script(tmp_path):
    """scripts/run_notebooks.py: sequential run with overrides; stops with exit code 1 at the first failing cell."""
    pytest.importorskip("nbclient")
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import run_notebooks
    raw, conv, models, out = tmp_path / "raw", tmp_path / "converted", tmp_path / "models", tmp_path / "out"
    make_raw_workbooks(str(raw), n_sausages=2, rows_per_device=60)
    common = [f"RAW_DIR = {str(raw)!r}", f"CONVERTED_DIR = {str(conv)!r}", f"MODELS_DIR = {str(models)!r}",
              "QUICK = True", "N_JOBS = 1"]
    args = sum((["--set", o] for o in common), []) + ["--out", str(out)]
    assert run_notebooks.main(["01", "02", "03"] + args) == 0
    assert (models / "manifest.json").exists() and len(list(out.glob("*.ipynb"))) == 3
    nb = json.load(open(out / "03_train_and_evaluate.ipynb"))
    assert any("injected-parameters" in c["metadata"].get("tags", []) for c in nb["cells"])
    # a broken setting: notebook 03 fails, 04 is not started
    assert run_notebooks.main(["03", "04", "--set", "FEATURE_SET = 'nope'"] + args) == 1
    nb = json.load(open(out / "03_train_and_evaluate.ipynb"))
    assert any(o.get("output_type") == "error" for c in nb["cells"] for o in c.get("outputs", []))



# ── row-level augmentation ──────────────────────────────────────────────────
def _recording(n_days=3, per_day=96):
    t = pd.date_range("2023-04-01", periods=n_days * per_day, freq=pd.Timedelta(days=1) / per_day, tz="UTC")
    n = len(t)
    df = pd.DataFrame({"timestamp": t, "sensor": "lse01", "lse01_conductivity": 200.0, "lse01_moisture": 5.0,
                       "lse01_temperature": 15.0, "ltcfs_temperature": 15.5, "ambient_temperature": 16.0})
    add_temperature_difference(df)
    return df[FEATURE_ORDER].assign(label=np.repeat(["Umroetung", "Reifung Phase 1", "Reifung Phase 2"], n // 3))


def test_calibration_is_constant_per_segment_and_keeps_labels():
    from ml_core.augmentation import augment_rows, copy_rng
    df = _recording()
    out = augment_rows(df, ["calibration_large"], copy_rng(0, 0))
    day = pd.to_datetime(out["timestamp"]).dt.floor("1D")
    assert out.groupby(day)["lse01_moisture"].nunique().eq(1).all()        # one offset per day
    assert out.groupby(day)["lse01_conductivity"].nunique().eq(1).all()
    assert out["lse01_moisture"].sub(5.0).abs().le(2.0 + 1e-9).all() and (out["lse01_moisture"] >= 0).all()
    ratio = out["lse01_conductivity"] / 200.0
    assert ratio.between(0.67 - 1e-9, 1.5 + 1e-9).all() and ratio.nunique() > 1
    assert (out["label"] == df["label"]).all()                              # rows stay aligned with their labels
    assert out["ltcfs_temperature"].eq(15.5).all()                          # calibration leaves temperatures alone


def test_robust_drops_uplinks_and_moves_probe_and_difference_together():
    from ml_core.augmentation import augment_rows, copy_rng
    df = _recording(n_days=10)
    out = augment_rows(df, ["robust"], copy_rng(1, 0))
    assert 0.75 < len(out) / len(df) < 0.95                                 # ~15 % of the uplinks lost
    air = out["ltcfs_temperature"] - out["ltcfs_temperature_diff"]           # noise on the probe, not on the air
    np.testing.assert_allclose(air, 16.0, atol=1e-9)
    assert out["lse01_moisture"].std() > 0 and (out["lse01_moisture"] >= 0).all()
    # same seed, same copy: reproducible regardless of worker order
    pd.testing.assert_frame_equal(out, augment_rows(df, ["robust"], copy_rng(1, 0)))


def test_augmented_copies_of_the_held_out_sausage_are_never_trained_on(synthetic_dir):
    w = load_windows(str(synthetic_dir / "converted"), n_jobs=1, log=_quiet)
    wa = w.with_augmentation(["calibration", "robust", "temperature"], n_jobs=1, log=_quiet)
    assert wa.augmentation == ("calibration", "robust") and all(len(c) == 2 for c in wa.augmented.values())
    sel = wa.select("temperature_difference").without_raw()
    held = sel.recordings[0]
    X, y, groups = sel.stack(exclude=held, augmented=True)
    assert held not in set(groups) and list(X.columns) == sel.columns
    n_orig = sum(len(sel.y[g]) for g in sel.recordings if g != held)
    assert len(y) > 2 * n_orig                                              # originals + two copies (minus dropout)


# ── gradient boosting as an optional fourth model ───────────────────────────
def test_gradient_boosting_trains_exports_and_explains(synthetic_dir, tmp_path):
    from demonstrator.explain import explain
    from demonstrator.inference import ModelRegistry
    w = load_windows(str(synthetic_dir / "converted"), n_jobs=1, log=_quiet)
    out = tmp_path / "models"
    train_and_evaluate(w, str(out), keys=("logreg", "hgb"), quick=True, n_jobs=1, log=_quiet)
    reg = ModelRegistry(str(out))
    assert reg.ok and reg.keys == ["logreg", "hgb"]
    rows = load_recording(str(synthetic_dir / "converted" / "series_1_sausage_1.csv"))[FEATURE_ORDER].iloc[200:210]
    ex = explain(reg, rows, None)
    d = ex["models"]["hgb"]["detail"]
    assert d["kind"] == "boosting" and d["stages"] and len(d["stages"][-1]["proba"]) == len(d["classes"])
    assert abs(sum(d["stages"][-1]["proba"]) - 1) < 1e-6


def test_hyperparameters_of_other_models_are_kept(tmp_path, synthetic_dir):
    w = load_windows(str(synthetic_dir / "converted"), n_jobs=1, log=_quiet).select("temperature_difference")
    save_hyperparameters({"rf": {"params": {"n_estimators": 7}, "score": 0.5}}, w, str(tmp_path))
    save_hyperparameters({"hgb": {"params": {"max_iter": 9}, "score": 0.6}}, w, str(tmp_path))
    params, _, fs = load_hyperparameters(str(tmp_path))
    assert params["rf"] == {"n_estimators": 7} and params["hgb"] == {"max_iter": 9} and fs == "temperature_difference"


# ── long-term context features ──────────────────────────────────────────────
def test_context_features_match_a_direct_computation(synthetic_dir):
    """Mean/std/trend over the last 6 h of a sensor's readings, at the window's last uplink; transform_last agrees."""
    df = load_recording(str(synthetic_dir / "converted" / "series_1_sausage_1.csv"))[FEATURE_ORDER]
    wf = WindowFeatures(10, 2, context=("6h",))
    X = wf.transform(df)
    i = 300
    k = i * 2 + 9                                                           # last row of window i
    past = df.iloc[:k + 1]
    s = past[(past["sensor"] == "lse01") & past["lse01_moisture"].notna()]
    t = pd.to_datetime(s["timestamp"])
    w = s[t > t.iloc[-1] - pd.Timedelta("6h")]
    days = (pd.to_datetime(w["timestamp"]) - t.iloc[0]).dt.total_seconds() / 86400
    assert X.loc[i, "lse01_moisture_ctxmean_6h"] == pytest.approx(w["lse01_moisture"].mean())
    assert X.loc[i, "lse01_moisture_ctxstd_6h"] == pytest.approx(w["lse01_moisture"].std())
    assert X.loc[i, "lse01_moisture_ctxtrend_6h"] == pytest.approx(np.polyfit(days, w["lse01_moisture"], 1)[0])
    last = wf.transform_last(past)
    np.testing.assert_allclose(last.iloc[0].to_numpy(float), X.loc[i, last.columns].to_numpy(float), equal_nan=True)
    assert feature_signal("lse01_moisture_ctxtrend_6h") == "lse01_moisture"


def test_models_with_context_run_in_the_demonstrator(synthetic_dir, tmp_path):
    """Train with context, then live, replay (animated and whole run), explanation and simulation must all work."""
    import time
    from fastapi.testclient import TestClient
    from demonstrator.app import create_app
    from demonstrator.config import Config, Device
    conv = str(synthetic_dir / "converted")
    w = load_windows(conv, context=("6h",), n_jobs=1, log=_quiet)
    assert any("_ctxtrend_6h" in c for c in w.columns)
    # windows without 6 h of history are left out of training and evaluation
    plain = load_windows(conv, n_jobs=1, log=_quiet)
    for g in w.recordings:
        start = pd.to_datetime(plain.raw[g]["timestamp"]).iloc[0]
        assert pd.to_datetime(w.times[g][0]) >= start + pd.Timedelta("6h") and len(w.X[g]) < len(plain.X[g])
        assert len(w.X[g]) == len(w.y[g]) == len(w.times[g])
    train_and_evaluate(w, str(tmp_path / "models"), keys=("logreg", "rf"), quick=True, n_jobs=1, log=_quiet)
    manifest = json.load(open(tmp_path / "models" / "manifest.json"))
    assert manifest["context"] == ["6h"] and any("ctx" in c for c in manifest["output_columns"])
    cfg = Config(models_dir=str(tmp_path / "models"), replay_dirs=[conv], recordings_dir=str(tmp_path / "rec"),
                 devices=[Device("A840410000000001", "W1", "lse01", "sausage_1")])
    with TestClient(create_app(cfg)) as c:
        for i in range(12):
            c.post("/ingest/uplink", json={"devEUI": "A840410000000001",
                                           "object": {"water_SOIL": 40 - i, "temp_SOIL": 15, "conduct_SOIL": 4000}})
        live = c.get("/api/state").json()["live"]["predictions"]["sausage_1"]
        assert live["ready"] and 0 <= live["history_real"] < 0.01          # seconds of real history, rest filled
        c.post("/api/replay/start", json={"group": "series_1_sausage_2", "speed": 200000, "start_fraction": 0.5})
        for _ in range(50):
            if c.get("/api/state").json()["replay"]["latest"]:
                break
            time.sleep(0.2)
        ex = c.get("/api/explain?source=replay").json()
        assert ex["ready"] and ex["context"] == ["6h"] and len(ex["uplinks"]) == 10
        assert any(f["feature"].endswith("_ctxtrend_6h") for f in ex["features"])
        c.post("/api/replay/start", json={"group": "series_1_sausage_2", "complete": True})
        for _ in range(100):
            st = c.get("/api/state").json()["replay"]["status"]
            if st["finished"] or st["error"]:
                break
            time.sleep(0.1)
        assert st["finished"] and not st["error"]
        sim = c.post("/api/simulate", json={"group": "series_1_sausage_1", "position": 0.7,
                                            "adjust": {"moisture_delta": -10}}).json()
        assert sim["ready"] and sim["simulated_means"]["lse01_moisture"] < sim["recorded_means"]["lse01_moisture"]


def test_fill_history_covers_the_horizon_with_the_earliest_stretch(synthetic_dir):
    from ml_core.windowing import fill_history
    df = load_recording(str(synthetic_dir / "converted" / "series_1_sausage_1.csv"))[FEATURE_ORDER]
    part = df.iloc[:40]
    filled, real = fill_history(part, pd.Timedelta("1D"), 10)
    ts = pd.to_datetime(filled["timestamp"], utc=True)
    span = pd.to_datetime(part["timestamp"], utc=True)
    assert ts.is_monotonic_increasing and ts.iloc[-1] - ts.iloc[0] > pd.Timedelta("23h")
    assert real == pytest.approx((span.iloc[-1] - span.iloc[0]) / pd.Timedelta("1D"))
    pd.testing.assert_frame_equal(filled.tail(40).reset_index(drop=True), part.reset_index(drop=True))  # real rows last
    assert set(filled["lse01_moisture"].dropna()) <= set(part["lse01_moisture"].dropna())              # only copies
    full, real_full = fill_history(df, pd.Timedelta("1D"), 10)
    assert real_full == 1.0 and len(full) == len(df)
