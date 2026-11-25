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
from ml_core.data import ALL_COLUMNS, FEATURE_ORDER, ROOT, add_temperature_difference
from ml_core.training import (FEATURE_SETS, QUICK_GRIDS, augment_temperature, compare_feature_sets, feature_signal,
                              load_hyperparameters, load_windows, save_hyperparameters, search_hyperparameters)
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
    variants = ["original", "temperature_difference", "temperature_difference+augmentation"]
    table, prog = compare_feature_sets(w, u, variants, keys=("logreg",), quick=True, log=_quiet)
    rho = table["series 2: phase order ρ"]
    assert rho[("original", "Logistic Regression")] < 0.8 < rho[("temperature_difference", "Logistic Regression")]
    assert rho[("temperature_difference+augmentation", "Logistic Regression")] > 0.8
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
