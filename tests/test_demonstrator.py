"""Demonstrator: decoders, live path, replay, explanation, simulation. Run: python -m pytest -q"""
from __future__ import annotations

import base64
import time

import pytest
from fastapi.testclient import TestClient

from demonstrator.app import create_app
from demonstrator.config import Config, Device
from demonstrator.decoders import decode_lse01, decode_values, decode_ws202, extract_envelope
from ml_core.data import FEATURE_ORDER, load_recording
from ml_core.training import DEFAULT_FEATURE_SET, DEMO_MODELS, set_signals


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


# ── end-to-end on synthetic data ────────────────────────────────────────────
@pytest.fixture
def client(trained, tmp_path):
    cfg = Config(models_dir=str(trained / "models"), replay_dirs=[str(trained / "converted")],
                 recordings_dir=str(tmp_path / "rec"),
                 devices=[Device("A840410000000001", "W1 LSE01", "lse01", "sausage_1"),
                          Device("A840410000000002", "W1 LTC", "ltcfs", "sausage_1"),
                          Device("24E1240000000000", "Env", "ws202", "environment")])
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
    assert pred["ready"] and set(pred["models"]) == set(DEMO_MODELS)
    for m in pred["models"].values():
        assert 0 <= m["confidence"] <= 1 and m["ms"] < 1000
    assert {d["type"] for d in st["live"]["devices"]} == {"lse01", "ltcfs", "ws202"}


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
    assert set(m["models"]) == set(DEMO_MODELS)
    assert "balanced_accuracy" in m["models"]["rf"] and m["agreement_on_held_out"]


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
        assert {o["signal"] for o in m["occlusion"]} == set(set_signals(DEFAULT_FEATURE_SET))  # default feature set
        assert abs(sum(m["proba"].values()) - 1) < 1e-6
    client.post("/api/replay/stop")


def test_explain_logreg_sum_is_exact(trained):
    """The weighted-sum view must reproduce the model's own decision score."""
    from demonstrator.explain import explain_model
    from demonstrator.inference import ModelRegistry
    reg = ModelRegistry(str(trained / "models"))
    pipe, _ = reg.model_for("logreg", None)
    rows = load_recording(str(trained / "converted" / "series_1_sausage_1.csv"))[FEATURE_ORDER].iloc[100:110]
    d = explain_model(pipe, "logreg", rows)["detail"]
    s = dict(pipe.steps)
    feats = s["windows"].transform(rows).tail(1)
    xs = s["scaler"].transform(s["imputer"].transform(feats))
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
    assert simm["ambient_temperature"] == pytest.approx(rec["ambient_temperature"])        # air stays as recorded
    assert simm["ltcfs_temperature_diff"] == pytest.approx(rec["ltcfs_temperature_diff"] + 10, abs=1e-6)
    ex = client.get("/api/explain?source=simulate").json()
    assert ex["ready"] and ex["source"] == "simulate" and ex["adjust"]["conductivity_factor"] == 0.5
    assert client.post("/api/simulate", json={"group": "nope"}).status_code == 400


def test_air_temperature_setting_drives_live_differences(client):
    for i in range(12):
        msg = [{"devEUI": "A840410000000001", "object": {"water_SOIL": 40, "temp_SOIL": 15, "conduct_SOIL": 3000}},
               {"devEUI": "A840410000000002", "object": {"TempC1": 16.0}},
               {"devEUI": "24E1240000000000", "object": {"pir": 0, "daylight": 1}}][i % 3]
        client.post("/ingest/uplink", json=msg)
    assert client.post("/api/ambient", json={"value": 99}).status_code == 400
    assert client.post("/api/ambient", json={"value": 12.5}).json()["value"] == 12.5
    st = client.get("/api/state").json()
    assert st["live"]["ambient_temperature"] == 12.5 and st["live"]["predictions"]["sausage_1"]["ready"]
    ex = client.get("/api/explain?source=live&slot=sausage_1").json()
    diff = next(f for f in ex["features"] if f["feature"] == "ltcfs_temperature_diff_mean")
    assert diff["value"] == pytest.approx(16.0 - 12.5)


def test_presentation_and_detail_pages(client):
    """'/' is the presentation view, '/details' the full dashboard; both load the shared helpers."""
    for path, marker in (("/", "simple.js"), ("/details", "app.js")):
        r = client.get(path)
        assert r.status_code == 200 and marker in r.text and "common.js" in r.text
    for f in ("simple.js", "common.js", "app.js"):
        assert client.get(f"/static/{f}").status_code == 200
    res = client.post("/api/simulate", json={"group": "series_1_sausage_1", "position": 0.5}).json()
    v = res["values"]                     # the measurement tiles of the presentation view
    assert {"lse01_moisture", "lse01_conductivity", "ltcfs_temperature", "ambient_temperature"} <= set(v)
    assert 0 < v["lse01_moisture"] < 100 and -30 < v["ambient_temperature"] < 50


# ── whole run at once ───────────────────────────────────────────────────────
def test_replay_whole_run_at_once(client):
    """complete=true predicts every window in one batch and ends in the same state as a finished replay."""
    r = client.post("/api/replay/start", json={"group": "series_1_sausage_2", "complete": True})
    assert r.status_code == 200, r.text
    for _ in range(100):
        rep = client.get("/api/state").json()["replay"]
        if rep["status"]["finished"] or rep["status"]["error"]:
            break
        time.sleep(0.1)
    st = rep["status"]
    assert st["finished"] and st["complete"] and st["progress"] == 1.0 and not st["error"]
    assert len(rep["history"]) > 50 and {"logreg", "rf", "mlp", "truth"} <= set(rep["history"][0])
    assert rep["running_score"]["n"] > 50 and all(0 <= rep["running_score"][k] <= 1 for k in ("logreg", "rf", "mlp"))
    assert rep["latest"]["ready"] and not rep["latest"]["any_model_saw_data"]   # fold models, as in the replay
    assert client.get("/api/explain?source=replay").json()["ready"]


def test_recordings_have_spans_for_readable_names(client):
    recs = {r["group"]: r for r in client.get("/api/recordings").json()}
    r = recs["series_1_sausage_1"]
    assert r["start"] < r["end"] and r["start"].startswith("20")


# ── persistent recording of case uplinks ────────────────────────────────────
def _uplinks(i):
    return [{"devEUI": "A840410000000001", "object": {"water_SOIL": 50 - i, "temp_SOIL": 15, "conduct_SOIL": 4000}},
            {"deviceInfo": {"devEui": "a840410000000002"}, "object": {"TempC1": 15.2}},
            {"devEUI": "24e1240000000000", "object": {"pir": 0, "daylight": 1}}][i % 3]


def test_uplinks_are_stored_and_restored_after_restart(trained, tmp_path):
    import glob
    import json
    rec_dir = tmp_path / "rec"
    cfg = Config(models_dir=str(trained / "models"), replay_dirs=[str(trained / "converted")],
                 recordings_dir=str(rec_dir),
                 devices=[Device("A840410000000001", "W1 LSE01", "lse01", "sausage_1"),
                          Device("A840410000000002", "W1 LTC", "ltcfs", "sausage_1"),
                          Device("24E1240000000000", "Env", "ws202", "environment")])
    with TestClient(create_app(cfg)) as c:
        for i in range(12):
            c.post("/ingest/uplink", json=_uplinks(i))
        c.post("/ingest/uplink", json={"devEUI": "FFFF000000000009", "object": {"x": 1}})        # unknown device
        c.post("/ingest/uplink?source=test", json=_uplinks(0))                                   # test: not stored
        before = c.get("/api/state").json()["live"]["predictions"]["sausage_1"]
    raw = [json.loads(line) for f in glob.glob(str(rec_dir / "raw_*.jsonl")) for line in open(f)]
    assert len(raw) == 13                                                     # every live uplink, incl. the unknown one
    assert any(r["uplink"].get("devEUI") == "FFFF000000000009" for r in raw)
    csvs = glob.glob(str(rec_dir / "live_*_sausage_1.csv"))
    assert len(csvs) == 1 and len(open(csvs[0]).read().splitlines()) == 1 + 12   # header + 12 decoded rows
    # restart: the recorded rows come back and the prediction is there without a new uplink
    with TestClient(create_app(cfg)) as c:
        after = c.get("/api/state").json()["live"]["predictions"]["sausage_1"]
    assert after["ready"] and {k: m["label"] for k, m in after["models"].items()} == \
        {k: m["label"] for k, m in before["models"].items()}


def test_agreement_counts_the_largest_group(client):
    st = None
    for i in range(12):
        client.post("/ingest/uplink", json=_uplinks(i))
    st = client.get("/api/state").json()["live"]["predictions"]["sausage_1"]
    labels = [m["label"] for m in st["models"].values()]
    assert st["agreement_count"] == max(labels.count(x) for x in labels)
    assert st["agreement"] == ("all" if st["agreement_count"] == len(labels) else "none" if st["agreement_count"] == 1 else "partial")
