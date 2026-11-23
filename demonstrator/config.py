"""Demonstrator configuration (TOML, stdlib only)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENSOR_TYPES = ("lse01", "ltcfs", "ws202")


@dataclass
class Device:
    dev_eui: str
    name: str
    type: str            # lse01 | ltcfs | ws202
    slot: str            # e.g. "sausage_1"; WS202 -> "environment"
    fields: dict = field(default_factory=dict)  # optional decoded-key overrides


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8000
    ingest_token: str = ""
    models_dir: str = os.path.join(ROOT, "data", "models")
    replay_dirs: list = field(default_factory=lambda: [
        os.path.join(ROOT, "data", "training", "converted"),
        os.path.join(ROOT, "data", "recordings")])
    recordings_dir: str = os.path.join(ROOT, "data", "recordings")
    record_live: bool = True
    mqtt_enabled: bool = False
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = "#"
    mqtt_username: str = ""
    mqtt_password: str = ""
    devices: list = field(default_factory=list)
    path: str = ""

    def device(self, dev_eui: str) -> Device | None:
        key = normalize_eui(dev_eui)
        for d in self.devices:
            if normalize_eui(d.dev_eui) == key:
                return d
        return None

    @property
    def slots(self) -> list[str]:
        return sorted({d.slot for d in self.devices if d.type != "ws202"})


def normalize_eui(eui: str | None) -> str:
    return (eui or "").replace("-", "").replace(":", "").strip().lower()


def _abs(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_config(path: str | None) -> Config:
    cfg = Config()
    if not path:
        return cfg
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    base = ROOT  # paths in the file are relative to the repository root
    srv, paths, mq = raw.get("server", {}), raw.get("paths", {}), raw.get("mqtt", {})
    cfg.path = os.path.abspath(path)
    cfg.host = srv.get("host", cfg.host)
    cfg.port = int(srv.get("port", cfg.port))
    cfg.ingest_token = srv.get("ingest_token", "")
    cfg.record_live = bool(srv.get("record_live", True))
    if "models" in paths:
        cfg.models_dir = _abs(paths["models"], base)
    if "replay_data" in paths:
        cfg.replay_dirs = [_abs(p, base) for p in paths["replay_data"]]
    if "recordings" in paths:
        cfg.recordings_dir = _abs(paths["recordings"], base)
    cfg.mqtt_enabled = bool(mq.get("enabled", False))
    cfg.mqtt_host = mq.get("host", cfg.mqtt_host)
    cfg.mqtt_port = int(mq.get("port", cfg.mqtt_port))
    cfg.mqtt_topic = mq.get("topic", cfg.mqtt_topic)
    cfg.mqtt_username = mq.get("username", "")
    cfg.mqtt_password = mq.get("password", "")
    for d in raw.get("devices", []):
        t = d["type"].lower().replace("-", "")
        if t not in SENSOR_TYPES:
            raise ValueError(f"device {d.get('dev_eui')}: type must be one of {SENSOR_TYPES}")
        cfg.devices.append(Device(dev_eui=d["dev_eui"], name=d.get("name", d["dev_eui"]), type=t,
                                  slot=d.get("slot", "environment" if t == "ws202" else "sausage_1"),
                                  fields=d.get("fields", {})))
    return cfg
