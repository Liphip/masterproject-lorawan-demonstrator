"""Demonstrator configuration (TOML, stdlib only)."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

from ml_core.data import CONVERTED_DIR, MODELS_DIR, RECORDINGS_DIR, ROOT

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
    models_dir: str = MODELS_DIR
    replay_dirs: list = field(default_factory=lambda: [CONVERTED_DIR, RECORDINGS_DIR])
    recordings_dir: str = RECORDINGS_DIR
    record_live: bool = True
    # The case has no air sensor; live temperature differences use this value (changeable in the UI).
    ambient_temperature: float = 20.0
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
    srv, paths = raw.get("server", {}), raw.get("paths", {})
    cfg.path = os.path.abspath(path)
    cfg.host = srv.get("host", cfg.host)
    cfg.port = int(srv.get("port", cfg.port))
    cfg.ingest_token = srv.get("ingest_token", "")
    cfg.record_live = bool(srv.get("record_live", True))
    cfg.ambient_temperature = float(raw.get("case", {}).get("ambient_temperature", cfg.ambient_temperature))
    if "models" in paths:
        cfg.models_dir = _abs(paths["models"], base)
    if "replay_data" in paths:
        cfg.replay_dirs = [_abs(p, base) for p in paths["replay_data"]]
    if "recordings" in paths:
        cfg.recordings_dir = _abs(paths["recordings"], base)
    for d in raw.get("devices", []):
        t = d["type"].lower().replace("-", "")
        if t not in SENSOR_TYPES:
            raise ValueError(f"device {d.get('dev_eui')}: type must be one of {SENSOR_TYPES}")
        cfg.devices.append(Device(dev_eui=d["dev_eui"], name=d.get("name", d["dev_eui"]), type=t,
                                  slot=d.get("slot", "environment" if t == "ws202" else "sausage_1"),
                                  fields=d.get("fields", {})))
    return cfg
