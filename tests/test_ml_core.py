"""ml_core: window features, conversion, search → training hand-off, notebooks."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from ml_core import ModelPipeline, WindowFeatures
from ml_core.conversion import classify_device, convert_all
from ml_core.data import ALL_COLUMNS, ROOT
from ml_core.training import (FEATURE_SETS, QUICK_GRIDS, compare_feature_sets, load_hyperparameters, load_windows,
                              save_hyperparameters, search_hyperparameters)
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
    assert classify_device("Umgebung Temperatur & Luftfeuchtigkeit") == (None, None)


def test_conversion_of_both_layouts(tmp_path):
    make_raw_workbooks(str(tmp_path / "raw"), n_sausages=2, rows_per_device=12)
    summary = convert_all(str(tmp_path / "raw"), str(tmp_path / "converted"), log=_quiet)
    assert sorted(summary["file"]) == [f"series_{s}_sausage_{k}.csv" for s in (1, 2) for k in (1, 2)]
    s1 = pd.read_csv(tmp_path / "converted" / "series_1_sausage_2.csv")
    s2 = pd.read_csv(tmp_path / "converted" / "series_2_sausage_1.csv")
    assert list(s1.columns) == ALL_COLUMNS + ["label"]
    assert list(s2.columns) == ALL_COLUMNS            # unlabeled run: no label column, but NOT empty
    for df in (s1, s2):
        assert df["sensor"].value_counts().to_dict() == {"lse01": 12, "ltcfs": 12, "ws202": 12}  # pH/ambient dropped
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
    assert man["feature_set"] == "moisture_conductivity"                 # default: no temperature, no timing
    assert man["output_columns"] and all(c.startswith(("lse01_conductivity", "lse01_moisture")) for c in man["output_columns"])
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
    assert w.select("moisture_conductivity").columns == [c for c in w.columns if c.startswith(("lse01_conductivity", "lse01_moisture"))]
    table, prog = compare_feature_sets(w, u, ["all", "moisture_conductivity"], keys=("logreg",), quick=True, log=_quiet)
    rho = table["series 2: phase order ρ"]
    assert rho[("all", "Logistic Regression")] < 0 < 0.5 < rho[("moisture_conductivity", "Logistic Regression")]
    assert set(prog[("all", "logreg")]) == set(u.recordings)
    assert set(FEATURE_SETS) == {"all", "no_temperature", "moisture_conductivity"}


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
