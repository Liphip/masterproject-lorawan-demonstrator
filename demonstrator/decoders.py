"""Uplink parsing: gateway JSON -> device id, radio info, measured values.

Preferred path: configure a payload codec for each device on the UG63 embedded
network server, so the uplink JSON already contains decoded fields. The binary
fallback decoders below follow the vendors' published payload layouts as far as
known to the author and MUST be checked against the device manuals / firmware
version before relying on them. LTC-FS has no binary fallback on purpose.
"""
from __future__ import annotations

import base64
from typing import Any

# decoded-object keys accepted per model column (first match wins, case-insensitive)
FIELD_CANDIDATES = {
    "lse01_conductivity": ["conduct_soil", "conductivity", "soil_conductivity", "ec"],
    "lse01_moisture": ["water_soil", "moisture", "soil_moisture"],
    "lse01_temperature": ["temp_soil", "soil_temperature", "temperature_soil"],
    "ltcfs_temperature": ["temp_channel1", "tempc1", "temp1", "temperature", "temp"],
    "ws202_light_status": ["daylight", "light_status", "light"],
    "ws202_pir_status": ["pir", "pir_status", "occupancy", "motion"],
}
COLUMNS_BY_TYPE = {
    "lse01": ["lse01_conductivity", "lse01_moisture", "lse01_temperature"],
    "ltcfs": ["ltcfs_temperature"],
    "ws202": ["ws202_light_status", "ws202_pir_status"],
}
_STATE_WORDS = {"light": 1, "dark": 0, "trigger": 1, "triggered": 1, "normal": 0,
                "not triggered": 0, "idle": 0, "true": 1, "false": 0}


class DecodeError(ValueError):
    pass


# ── binary fallbacks ────────────────────────────────────────────────────────
def decode_ws202(b: bytes) -> dict:
    """Milesight WS202 (channel/type TLV): 01 75 battery, 03 00 PIR, 04 00 daylight."""
    out, i = {}, 0
    while i + 2 < len(b):  # each known record is 3 bytes
        ch, ty = b[i], b[i + 1]
        i += 2
        if (ch, ty) == (0x01, 0x75):
            out["battery"] = b[i]; i += 1
        elif (ch, ty) == (0x03, 0x00):
            out["pir"] = b[i]; i += 1
        elif (ch, ty) == (0x04, 0x00):
            out["daylight"] = b[i]; i += 1
        else:
            break  # unknown channel: stop instead of guessing lengths
    if "pir" not in out and "daylight" not in out:
        raise DecodeError("WS202 payload contains no PIR/daylight channel")
    return out


def decode_lse01(b: bytes) -> dict:
    """Dragino LSE01 (11 bytes): bat, DS18B20, moisture/100, soil temp/100, EC uS/cm."""
    if len(b) < 10:
        raise DecodeError(f"LSE01 payload too short ({len(b)} bytes)")
    s16 = lambda hi, lo: int.from_bytes(bytes([hi, lo]), "big", signed=True)
    return {
        "bat_v": ((b[0] << 8 | b[1]) & 0x3FFF) / 1000,
        "temp_ds18b20": s16(b[2], b[3]) / 10,
        "water_soil": (b[4] << 8 | b[5]) / 100,
        "temp_soil": s16(b[6], b[7]) / 100,
        "conduct_soil": b[8] << 8 | b[9],
    }


BINARY_DECODERS = {"ws202": decode_ws202, "lse01": decode_lse01}


# ── generic JSON handling ───────────────────────────────────────────────────
def _get(d: Any, *path):
    for p in path:
        if isinstance(d, list):
            d = d[0] if d else None
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def extract_envelope(msg: dict) -> dict:
    """Pull device id, time, radio info, payload from UG63 / ChirpStack v3+v4 / TTS JSON."""
    eui = (msg.get("devEUI") or msg.get("devEui") or msg.get("deveui") or msg.get("dev_eui")
           or _get(msg, "deviceInfo", "devEui") or _get(msg, "end_device_ids", "dev_eui"))
    name = msg.get("deviceName") or _get(msg, "deviceInfo", "deviceName") or \
        _get(msg, "end_device_ids", "device_id")
    rx = msg.get("rxInfo") or _get(msg, "uplink_message", "rx_metadata") or []
    rssi = _get(rx, "rssi")
    snr = _get(rx, "loRaSNR") if _get(rx, "loRaSNR") is not None else _get(rx, "snr")
    decoded = msg.get("object") or msg.get("decoded") or _get(msg, "uplink_message", "decoded_payload")
    if decoded is None:
        # UG63 can also put codec output at top level next to the envelope
        decoded = {k: v for k, v in msg.items()
                   if any(k.lower() in c for c in FIELD_CANDIDATES.values())} or None
    data = msg.get("data") or _get(msg, "uplink_message", "frm_payload")
    return {
        "dev_eui": eui, "device_name": name,
        "time": msg.get("time") or msg.get("received_at") or _get(rx, "time"),
        "fcnt": msg.get("fCnt") if msg.get("fCnt") is not None else _get(msg, "uplink_message", "f_cnt"),
        "fport": msg.get("fPort") if msg.get("fPort") is not None else _get(msg, "uplink_message", "f_port"),
        "rssi": rssi, "snr": snr, "decoded": decoded, "data_b64": data,
    }


def _num(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        w = v.strip().lower()
        if w in _STATE_WORDS:
            return float(_STATE_WORDS[w])
        try:
            return float(w)
        except ValueError:
            return None
    return None


def map_values(sensor_type: str, decoded: dict, overrides: dict | None = None) -> dict:
    """Map a decoded object to model columns for one sensor type."""
    lower = {str(k).lower(): v for k, v in (decoded or {}).items()}
    out = {}
    for col in COLUMNS_BY_TYPE[sensor_type]:
        keys = ([overrides[col]] if overrides and col in overrides else []) + FIELD_CANDIDATES[col]
        for k in keys:
            if k.lower() in lower:
                v = _num(lower[k.lower()])
                if v is not None:
                    out[col] = v
                    break
    return out


def decode_values(sensor_type: str, env: dict, overrides: dict | None = None) -> tuple[dict, str]:
    """Return (values, method) where method is 'codec' or 'builtin'."""
    if sensor_type == "generic":
        vals = {str(k): _num(v) for k, v in (env.get("decoded") or {}).items()
                if not isinstance(v, (dict, list)) and _num(v) is not None}
        if vals:
            return vals, "codec"
        raise DecodeError("generic device: needs a payload codec on the gateway")
    if env.get("decoded"):
        vals = map_values(sensor_type, env["decoded"], overrides)
        if vals:
            return vals, "codec"
    if env.get("data_b64") and sensor_type in BINARY_DECODERS:
        try:
            raw = base64.b64decode(env["data_b64"])
        except Exception as e:  # noqa: BLE001
            raise DecodeError(f"payload is not base64: {e}") from e
        vals = map_values(sensor_type, BINARY_DECODERS[sensor_type](raw), overrides)
        if vals:
            return vals, "builtin"
    raise DecodeError(f"no usable values for {sensor_type}; configure a payload codec on the gateway")
