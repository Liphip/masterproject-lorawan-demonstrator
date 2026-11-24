"""End-to-end and unit tests. Run from the repo root:  python -m pytest -q"""
from __future__ import annotations

import base64
import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402

from demonstrator.app import create_app  # noqa: E402
from demonstrator.config import Config, Device  # noqa: E402
from demonstrator.decoders import (decode_lse01, decode_values, decode_ws202,  # noqa: E402
                                   extract_envelope)
from ml_app import WindowingMethod  # noqa: E402
from ml_app.data import FEATURE_ORDER  # noqa: E402
from synthetic_data import make_dataset  # noqa: E402


# ── decoders ────────────────────────────────────────────────────────────────
def test_ws202_tlv():
    out = decode_ws202(bytes([0x01, 0x75, 0x5C, 0x03, 0x00, 0x01, 0x04, 0x00, 0x00]))
    assert out == {"battery": 92, "pir": 1, "daylight": 0}


def test_lse01_layout():
    # bat 3.3V, DS18B20 21.5C, moisture 45.67%, soil temp -1.25C, EC 1234
    soil_t = (-125) & 0xFFFF
    b = bytes([0x0C, 0xE4, 0x00, 0xD7, 0x11, 0xD7, soil_t >> 8, soil_t & 0xFF, 0x04, 0xD2, 0x00])
    out = decode_lse01(b)
    assert out["bat_v"] == pytest.approx(3.3, abs=0.01)
    assert out["temp_ds18b20"] == pytest.approx(21.5)
    assert out["water_soil"] == pytest.approx(45.67)
    assert out["temp_soil"] == pytest.approx(-1.25)
    assert out["conduct_soil"] == 1234


def test_envelopes_and_codec_mapping():
    v4 = {"deviceInfo": {"devEui": "a840410000000001", "deviceName": "w1"},
          "rxInfo": [{"rssi": -80, "snr": 7.5}], "object": {"TempC1": "12.5"}}
    env = extract_envelope(v4)
    assert env["dev_eui"] == "a840410000000001" and env["rssi"] == -80 and env["snr"] == 7.5
    assert decode_values("ltcfs", env)[0] == {"ltcfs_temperature": 12.5}
    ug = {"devEUI": "24e1240000000000", "rxInfo": [{"rssi": -60, "loRaSNR": 9}],
          "data": base64.b64encode(bytes([0x03, 0x00, 0x00, 0x04, 0x00, 0x01])).decode()}
    vals, method = decode_values("ws202", extract_envelope(ug))
    assert vals == {"ws202_pir_status": 0.0, "ws202_light_status": 1.0} and method == "builtin"
    assert decode_values("ws202", {"decoded": {"pir": "trigger", "daylight": "dark"}})[0] == \
        {"ws202_pir_status": 1.0, "ws202_light_status": 0.0}


# ── windowing fixes ─────────────────────────────────────────────────────────
def _rows():
    t = pd.date_range("2024-01-01", periods=10, freq="60s", tz="UTC")
    return pd.DataFrame({"timestamp": t,
                         "sensor": ["ws202", "lse01", "ltcfs"] * 3 + ["ws202"],
                         "lse01_conductivity": [np.nan, 100, np.nan, np.nan, 110, np.nan, np.nan, 130, np.nan, np.nan],
                         "lse01_moisture": np.nan, "lse01_temperature": np.nan,
                         "ltcfs_temperature": [np.nan, np.nan, 10, np.nan, np.nan, 11, np.nan, np.nan, 13, np.nan]})


def test_slope_uses_valid_readings_and_resolution():
    df = _rows()
    out_ns = WindowingMethod(window_size=10, stride=2).transform(df)
    assert out_ns["lse01_conductivity_slope"].iloc[0] == pytest.approx(30 / 6)
    assert out_ns["ltcfs_temperature_slope"].iloc[0] == pytest.approx(3 / 6)
    df_us = df.copy()
    df_us["timestamp"] = df_us["timestamp"].astype("datetime64[us, UTC]")
    out_us = WindowingMethod(window_size=10, stride=2).transform(df_us)
    assert out_us["timestamp_duration"].iloc[0] == pytest.approx(540.0)
    assert out_ns["timestamp_duration"].iloc[0] == pytest.approx(540.0)


def test_output_columns_fixed_schema():
    df = _rows()
    df = df[df["sensor"] != "ltcfs"]  # a sensor missing from the window
    cols = ["lse01_conductivity_mean", "ltcfs_temperature_mean", "timestamp_interval"]
    out = WindowingMethod(window_size=5, stride=1, output_columns=cols).transform(df)
    assert list(out.columns) == cols and out["ltcfs_temperature_mean"].isna().all()


# ── end-to-end on synthetic data ────────────────────────────────────────────
@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    base = tmp_path_factory.mktemp("demo")
    data = base / "converted"
    make_dataset(str(data), n_sausages=3, days=2)
    sys.path.insert(0, os.path.join(ROOT, "training"))
    import train_models
    train_models.main(["--data-dir", str(data), "--out", str(base / "models"), "--quick"])
    return base


@pytest.fixture
def client(trained, tmp_path):
    cfg = Config(models_dir=str(trained / "models"), replay_dirs=[str(trained / "converted")],
                 recordings_dir=str(tmp_path / "rec"),
                 devices=[Device("A840410000000001", "W1 LSE01", "lse01", "sausage_1"),
                          Device("A840410000000002", "W1 LTC", "ltcfs", "sausage_1"),
                          Device("24E1240000000000", "Env", "ws202", "environment"),
                          Device("A84041000000000C", "S31", "generic", "environment")])
    with TestClient(create_app(cfg)) as c:
        yield c


def test_live_path_produces_three_predictions(client):
    for i in range(12):
        kind = i % 3
        msg = [{"devEUI": "A840410000000001", "rxInfo": [{"rssi": -70, "loRaSNR": 8}],
                "object": {"water_SOIL": 50 - i, "temp_SOIL": 15, "conduct_SOIL": 4000}},
               {"deviceInfo": {"devEui": "a8-40-41-00-00-00-00-02"}, "object": {"TempC1": 15.2}},
               {"devEUI": "24e1240000000000", "object": {"pir": 0, "daylight": 1}}][kind]
        r = client.post("/ingest/uplink", json=msg)
        assert r.status_code == 200 and r.json()["results"][0]["mapped"], r.text
    st = client.get("/api/state").json()
    pred = st["live"]["predictions"]["sausage_1"]
    assert pred["ready"] and set(pred["models"]) == {"logreg", "rf", "mlp"}
    for m in pred["models"].values():
        assert 0 <= m["confidence"] <= 1 and m["ms"] < 1000
    assert {d["type"] for d in st["live"]["devices"]} == {"lse01", "ltcfs", "ws202", "generic"}


def test_unknown_device_is_listed_not_mapped(client):
    r = client.post("/ingest/uplink", json={"devEUI": "FFFF000000000001", "object": {"x": 1}})
    assert r.json()["results"][0]["mapped"] is False
    devs = client.get("/api/state").json()["live"]["devices"]
    assert any(not d["configured"] for d in devs)


def test_replay_uses_held_out_fold_models(client):
    groups = {r["group"] for r in client.get("/api/recordings").json()}
    assert "series_1_sausage_2" in groups and "series_2_sausage_1" in groups
    r = client.post("/api/replay/start", json={"group": "series_1_sausage_2", "speed": 200000})
    assert r.status_code == 200, r.text
    for _ in range(60):
        rep = client.get("/api/state").json()["replay"]
        if rep["latest"] and rep["running_score"].get("n", 0) > 5:
            break
        time.sleep(0.2)
    latest = rep["latest"]
    assert latest and latest["truth"] is not None
    assert not latest["any_model_saw_data"], "fold model should not have seen this sausage"
    client.post("/api/replay/stop")
    # unlabeled independent run: final models, no ground truth
    client.post("/api/replay/start", json={"group": "series_2_sausage_1", "speed": 200000})
    for _ in range(60):
        rep = client.get("/api/state").json()["replay"]
        if rep["latest"]:
            break
        time.sleep(0.2)
    assert rep["latest"]["truth"] is None and not rep["latest"]["any_model_saw_data"]
    client.post("/api/replay/stop")


def test_metrics_endpoint(client):
    m = client.get("/api/metrics").json()
    assert set(m["models"]) == {"logreg", "rf", "mlp"}
    assert "balanced_accuracy" in m["models"]["rf"] and m["agreement_on_held_out"]


def test_generic_device_is_display_only(client):
    r = client.post("/ingest/uplink", json={"devEUI": "A84041000000000C",
                                            "object": {"TempC_SHT": 21.3, "Hum_SHT": "48.5", "Work_mode": "IIC"}})
    assert r.json()["results"][0]["values"] == {"TempC_SHT": 21.3, "Hum_SHT": 48.5}
    assert "environment" not in client.get("/api/state").json()["live"]["slots"]


def test_explain_replay_snapshot(client):
    client.post("/api/replay/start", json={"group": "series_1_sausage_1", "speed": 50000, "start_fraction": 0.5})
    for _ in range(50):
        if client.get("/api/state").json()["replay"]["latest"]:
            break
        time.sleep(0.2)
    client.post("/api/replay/pause", json={"paused": True})
    assert client.get("/api/state").json()["replay"]["status"]["paused"]
    ex = client.get("/api/explain?source=replay").json()
    assert ex["ready"] and len(ex["uplinks"]) == 10 and ex["features"]
    lr, rf, mlp = ex["models"]["logreg"], ex["models"]["rf"], ex["models"]["mlp"]
    assert lr["detail"]["kind"] == "weighted_sum" and lr["detail"]["top"]
    assert sum(rf["detail"]["votes"]) == rf["detail"]["n_trees"]
    assert mlp["detail"]["layers"][0]["size"] == mlp["n_features_used"]
    for m in ex["models"].values():
        assert {o["signal"] for o in m["occlusion"]} >= {"lse01_moisture", "timestamp"}
        assert abs(sum(m["proba"].values()) - 1) < 1e-6
    client.post("/api/replay/stop")


def test_explain_logreg_sum_is_exact(trained):
    """The weighted-sum view must reproduce the model's own decision score."""
    import numpy as np
    from demonstrator.explain import explain_model
    from demonstrator.inference import ModelRegistry
    from ml_app.data import load_converted_csv
    reg = ModelRegistry(str(trained / "models"))
    pipe, _ = reg.model_for("logreg", None)
    rows = load_converted_csv(str(trained / "converted" / "series_1_sausage_1.csv"))[FEATURE_ORDER].iloc[100:110]
    d = explain_model(pipe, "logreg", rows)["detail"]
    s = dict(pipe.steps)
    feats = s["windowing"].transform(rows).tail(1)
    xs = s["scaler"].transform(s["inputer"].transform(feats))
    clf = pipe.steps[-1][1]
    k = list(clf.classes_).index(explain_model(pipe, "logreg", rows)["label"])
    expected = float(clf.estimators_[k].decision_function(xs)[0])
    assert d["score"] == pytest.approx(expected, abs=1e-6)
    assert sum(t["contribution"] for t in d["top"]) + d["rest"] + d["intercept"] == pytest.approx(expected, abs=1e-6)


def test_virtual_sausage_simulation(client):
    base = client.post("/api/simulate", json={"group": "series_1_sausage_1", "position": 0.1}).json()
    assert base["ready"] and base["simulated"] and base["truth"] is not None
    assert not base["truth_is_for_unmodified"]
    sim = client.post("/api/simulate", json={"group": "series_1_sausage_1", "position": 0.1,
                                             "adjust": {"moisture_delta": -20, "conductivity_factor": 0.5,
                                                        "temperature_delta": 99}}).json()
    assert sim["adjust"]["temperature_delta"] == 10.0  # clipped to the slider range
    assert sim["truth_is_for_unmodified"]
    rec, simm = sim["recorded_means"], sim["simulated_means"]
    assert simm["lse01_moisture"] == pytest.approx(rec["lse01_moisture"] - 20, abs=1e-6)
    assert simm["lse01_conductivity"] == pytest.approx(rec["lse01_conductivity"] * 0.5, rel=1e-6)
    assert simm["ltcfs_temperature"] == pytest.approx(rec["ltcfs_temperature"] + 10, abs=1e-6)
    ex = client.get("/api/explain?source=simulate").json()
    assert ex["ready"] and ex["source"] == "simulate" and ex["adjust"]["conductivity_factor"] == 0.5
    assert client.post("/api/simulate", json={"group": "nope"}).status_code == 400
