"""Live LoRaWAN path: normalise uplinks, keep device state, record, run models."""
from __future__ import annotations

import csv
import datetime as dt
import glob
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
# After a restart, rows younger than this are loaded back from the recordings, so predictions continue at once.
RESTORE_MAX_AGE = dt.timedelta(hours=6)   # at least; longer if the models use context features (e.g. 7 days)
# Rows kept per slot / of the environment sensor: enough for 7 days of context at one uplink per sensor and minute
# would be ~20k; the case sensors send every few minutes, so this covers a week comfortably.
SLOT_ROWS, ENV_ROWS = 6000, 3000


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
        self.env_rows = deque(maxlen=ENV_ROWS)
        self.predictions: dict[str, dict] = {}
        self.ambient = cfg.ambient_temperature
        self.uplinks = 0
        self.errors = deque(maxlen=20)
        self.last_uplink_at: str | None = None
        self._file_lock = threading.Lock()  # uplinks arrive in parallel threads; one writer per file at a time
        for d in cfg.devices:
            self.devices[normalize_eui(d.dev_eui)] = self._device_entry(d.dev_eui, d.name, d.type, d.slot, True)
        for s in cfg.slots:
            self.slot_rows[s] = deque(maxlen=SLOT_ROWS)
        self._restore()

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
        if self.cfg.record_live and source == "live":
            self._record_raw(now, msg)  # every uplink, before anything can fail (unknown devices, decode errors)
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
                self.slot_rows.setdefault(dev_cfg.slot, deque(maxlen=SLOT_ROWS)).append(row)
                affected = [dev_cfg.slot]
        if self.cfg.record_live and source == "live":
            self._record(row, affected)
        for slot in affected:
            self._predict(slot)
        return {"accepted": True, "mapped": True, "values": values}

    def _error(self, text):
        """Remember an error for the dashboard's error list."""
        with self.lock:
            self.errors.appendleft({"time": utcnow().isoformat(), "error": text})

    # ── recording (same schema as data/converted → replayable) ──────────────
    def _record_raw(self, now: dt.datetime, msg: dict):
        """Append the uplink exactly as received to data/recordings/raw_<day>.jsonl (all devices, also unknown ones
        and those that fail to decode). The source of truth: CSVs can be rebuilt from it if a decoder was wrong."""
        try:
            os.makedirs(self.cfg.recordings_dir, exist_ok=True)
            path = os.path.join(self.cfg.recordings_dir, f"raw_{now:%Y%m%d}.jsonl")
            line = json.dumps({"received": now.isoformat(), "uplink": msg}, ensure_ascii=False) + "\n"
            with self._file_lock:
                _append(path, line)
        except (OSError, TypeError, ValueError) as e:
            self._error(f"recording failed: {e}")

    def _record(self, row: dict, slots: list[str]):
        """Append a decoded uplink to data/recordings/live_<day>_<slot>.csv (training-data schema, so it can be
        replayed). WS202 rows go to every slot's file, as in the training data."""
        try:
            os.makedirs(self.cfg.recordings_dir, exist_ok=True)
            day = row["timestamp"].strftime("%Y%m%d")
            for slot in slots:
                p = os.path.join(self.cfg.recordings_dir, f"live_{day}_{slot}.csv")
                with self._file_lock:
                    new = not os.path.exists(p) or os.path.getsize(p) == 0
                    with open(p, "a", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=ALL_COLUMNS, extrasaction="ignore")
                        if new:
                            w.writeheader()
                        w.writerow({**row, "timestamp": row["timestamp"].isoformat()})
                        fh.flush()
                        os.fsync(fh.fileno())
        except OSError as e:
            self._error(f"recording failed: {e}")

    def _restore(self):
        """Load the latest recorded rows of every slot (younger than RESTORE_MAX_AGE) after a restart.

        Rows come from the live_<day>_<slot>.csv files of the days concerned (the models' context horizon, at
        least RESTORE_MAX_AGE); the WS202 rows in them refill the shared environment rows.
        Devices themselves reappear with their next uplink.
        """
        cutoff = pd.Timestamp(utcnow() - max(RESTORE_MAX_AGE, self.models.history.to_pytimedelta()
                                             if self.models.ok else RESTORE_MAX_AGE))
        env = {}
        for slot in list(self.slot_rows):
            files = sorted(glob.glob(os.path.join(self.cfg.recordings_dir, f"live_*_{slot}.csv")))
            day0 = f"live_{cutoff:%Y%m%d}"
            files = [f for f in files if os.path.basename(f) >= day0]   # only the days that can hold recent rows
            parts = []
            for f in files:
                try:
                    parts.append(pd.read_csv(f, parse_dates=["timestamp"]))
                except (OSError, ValueError, pd.errors.ParserError) as e:
                    self._error(f"could not restore {os.path.basename(f)}: {e}")
            if not parts:
                continue
            df = pd.concat(parts, ignore_index=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[df["timestamp"] >= cutoff]
            for r in df.to_dict("records"):
                r = {k: v for k, v in r.items() if not (isinstance(v, float) and pd.isna(v))}
                if r.get("sensor") == "ws202":
                    env[r["timestamp"]] = r
                else:
                    self.slot_rows[slot].append(r)
        for t in sorted(env):
            self.env_rows.append(env[t])
        restored = sum(len(v) for v in self.slot_rows.values())
        if restored:
            log.info("restored %d recorded rows from %s", restored, self.cfg.recordings_dir)
            for slot in list(self.slot_rows):
                self._predict(slot)

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


def _append(path: str, text: str):
    """Append text and force it to disk (an uplink arrives about once a minute, so fsync costs nothing)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
