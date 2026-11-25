"""Live LoRaWAN path: normalise uplinks, keep device state, record, run models."""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import threading
from collections import deque

import pandas as pd

from ml_core.data import ALL_COLUMNS, FEATURE_ORDER, add_temperature_difference

from .config import Config, normalize_eui
from .decoders import DecodeError, decode_values, extract_envelope
from .inference import ModelRegistry

log = logging.getLogger(__name__)


def utcnow() -> dt.datetime:
    """Current time, timezone-aware UTC."""
    return dt.datetime.now(dt.timezone.utc)


class LiveState:
    """State of the live LoRaWAN path: devices seen, recent uplinks, rows per sausage slot, predictions.

    Thread-safe (one lock): uplinks arrive in worker threads, the dashboard reads snapshots. The WS202 rows are shared
    by all slots, as in the training data (one environment sensor for all sausages).
    """
    def __init__(self, cfg: Config, models: ModelRegistry):
        """Pre-register the configured devices and slots, so they show up before their first uplink."""
        self.cfg, self.models = cfg, models
        self.lock = threading.Lock()
        self.devices: dict[str, dict] = {}
        self.recent = deque(maxlen=40)
        self.slot_rows: dict[str, deque] = {}
        self.env_rows = deque(maxlen=200)
        self.predictions: dict[str, dict] = {}
        self.ambient = cfg.ambient_temperature
        self.uplinks = 0
        self.errors = deque(maxlen=20)
        self.last_uplink_at: str | None = None
        for d in cfg.devices:
            self.devices[normalize_eui(d.dev_eui)] = self._device_entry(d.dev_eui, d.name, d.type, d.slot, True)
        for s in cfg.slots:
            self.slot_rows[s] = deque(maxlen=500)

    @staticmethod
    def _device_entry(eui, name, typ, slot, configured):
        """Initial dashboard entry for a device."""
        return {"dev_eui": eui, "name": name, "type": typ, "slot": slot, "configured": configured,
                "last_seen": None, "uplinks": 0, "rssi": None, "snr": None, "values": {},
                "decode_method": None, "last_error": None}

    # ── ingest ──────────────────────────────────────────────────────────────
    def handle_uplink(self, msg: dict, source: str = "live") -> dict:
        """Process one uplink from the gateway: identify the device, decode, record, predict.

        Unknown devices are shown (so a missing config entry is visible) but not used. Returns a short status that is
        sent back to the gateway / test script.
        """
        now = utcnow()
        env = extract_envelope(msg)
        eui = normalize_eui(env["dev_eui"])
        if not eui:
            self._error("uplink without DevEUI ignored")
            return {"accepted": False, "reason": "no DevEUI"}
        dev_cfg = self.cfg.device(eui)
        with self.lock:
            dev = self.devices.get(eui)
            if dev is None:
                dev = self._device_entry(env["dev_eui"], env["device_name"] or "unknown device",
                                         "unknown", "-", False)
                self.devices[eui] = dev
            dev.update(last_seen=now.isoformat(), rssi=env["rssi"], snr=env["snr"])
            dev["uplinks"] += 1
            self.uplinks += 1
            self.last_uplink_at = now.isoformat()
        entry = {"time": now.isoformat(), "dev_eui": env["dev_eui"], "name": dev["name"],
                 "type": dev["type"], "rssi": env["rssi"], "snr": env["snr"], "fcnt": env["fcnt"],
                 "source": source, "values": {}, "error": None}
        if dev_cfg is None:
            entry["error"] = "device not in config (add it to config/demo.toml)"
            with self.lock:
                dev["values"] = env["decoded"] or {}
                self.recent.appendleft(entry)
            return {"accepted": True, "mapped": False}
        try:
            values, method = decode_values(dev_cfg.type, env, dev_cfg.fields)
        except DecodeError as e:
            entry["error"] = str(e)
            with self.lock:
                dev["last_error"] = str(e)
                self.recent.appendleft(entry)
            self._error(f"{dev_cfg.name}: {e}")
            return {"accepted": True, "mapped": False, "error": str(e)}
        entry["values"] = values
        row = {"timestamp": pd.Timestamp(now), "sensor": dev_cfg.type, **values}
        with self.lock:
            dev.update(values=values, decode_method=method, last_error=None)
            self.recent.appendleft(entry)
            if dev_cfg.type == "ws202":
                self.env_rows.append(row)
                affected = list(self.slot_rows)
            else:
                self.slot_rows.setdefault(dev_cfg.slot, deque(maxlen=500)).append(row)
                affected = [dev_cfg.slot]
        if self.cfg.record_live and source == "live":
            self._record(row, affected, msg)
        for slot in affected:
            self._predict(slot)
        return {"accepted": True, "mapped": True, "values": values}

    def _error(self, text):
        """Remember an error for the dashboard's error list."""
        with self.lock:
            self.errors.appendleft({"time": utcnow().isoformat(), "error": text})

    # ── recording (same schema as data/converted → replayable) ──────────────
    def _record(self, row: dict, slots: list[str], raw: dict):
        """Append a decoded uplink to data/recordings/live_<day>_<slot>.csv (training-data schema, so it can be
        replayed) and the raw uplink to raw_<day>.jsonl (to check decoders later).
        """
        try:
            os.makedirs(self.cfg.recordings_dir, exist_ok=True)
            day = row["timestamp"].strftime("%Y%m%d")
            for slot in slots:
                p = os.path.join(self.cfg.recordings_dir, f"live_{day}_{slot}.csv")
                new = not os.path.exists(p)
                with open(p, "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    w.writerow({**row, "timestamp": row["timestamp"].isoformat()})
            with open(os.path.join(self.cfg.recordings_dir, f"raw_{day}.jsonl"), "a") as fh:
                fh.write(json.dumps({"received": row["timestamp"].isoformat(), "uplink": raw}) + "\n")
        except OSError as e:
            self._error(f"recording failed: {e}")

    # ── inference ───────────────────────────────────────────────────────────
    def window_rows(self, slot: str) -> pd.DataFrame:
        """The latest rows of a slot plus the environment rows, in time order, with the temperature difference to
        the configured air temperature (the case has no air sensor).
        """
        with self.lock:
            rows = list(self.slot_rows.get(slot, [])) + list(self.env_rows)
        if not rows:
            return pd.DataFrame(columns=FEATURE_ORDER)
        df = pd.DataFrame(rows).reindex(columns=FEATURE_ORDER).sort_values("timestamp", kind="stable")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return add_temperature_difference(df, ambient=self.ambient)[FEATURE_ORDER].reset_index(drop=True)

    def set_ambient(self, value: float):
        """Air temperature around the case (no sensor); re-predicts every slot."""
        with self.lock:
            self.ambient = float(value)
            slots = list(self.slot_rows)
        for slot in slots:
            self._predict(slot)

    def _predict(self, slot: str):
        """Predict the phase for a slot from its latest window; errors go to the dashboard, not to the gateway."""
        if not self.models.ok:
            return
        try:
            res = self.models.predict(self.window_rows(slot), group=None)
        except Exception as e:  # noqa: BLE001
            log.exception("live prediction failed")
            self._error(f"prediction failed for {slot}: {e!r}")
            return
        res["time"] = utcnow().isoformat()
        with self.lock:
            self.predictions[slot] = res

    # ── snapshot for the dashboard ──────────────────────────────────────────
    def snapshot(self) -> dict:
        """Copy of the live state for /api/state."""
        with self.lock:
            return {
                "ambient_temperature": self.ambient,
                "devices": sorted(self.devices.values(), key=lambda d: (not d["configured"], d["slot"], d["name"])),
                "recent": list(self.recent)[:25],
                "slots": sorted(set(self.slot_rows)),
                "predictions": dict(self.predictions),
                "uplinks": self.uplinks, "last_uplink_at": self.last_uplink_at,
                "errors": list(self.errors)[:10],
            }
