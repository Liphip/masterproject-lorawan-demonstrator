"""Send fake uplinks to a running demonstrator (no hardware needed).

They are tagged source=test, are NOT written to the recordings, and are labeled
"test" in the uplink log. Use them to check the HTTP path, the device mapping
and the dashboard before the gateway is set up.

    python scripts/send_test_uplink.py                       # one round, devices from config/demo.toml
    python scripts/send_test_uplink.py --rounds 12 --interval 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from demonstrator.config import load_config  # noqa: E402
from ml_core.data import ROOT  # noqa: E402


def payload(dev, i):
    if dev.type == "lse01":
        obj = {"water_SOIL": round(45 - 0.2 * i + random.uniform(-.5, .5), 2),
               "temp_SOIL": round(14 + random.uniform(-.3, .3), 2),
               "conduct_SOIL": int(3500 - 10 * i + random.uniform(-50, 50))}
    elif dev.type == "ltcfs":
        obj = {"TempC1": round(14.3 + random.uniform(-.3, .3), 2)}
    else:
        obj = {"pir": random.random() < .1, "daylight": "light"}
    return {"devEUI": dev.dev_eui, "deviceName": dev.name, "fCnt": i, "fPort": 2,
            "rxInfo": [{"rssi": random.randint(-95, -55), "loRaSNR": round(random.uniform(2, 10), 1)}],
            "object": obj}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "demo.toml"))
    ap.add_argument("--url", default="http://127.0.0.1:8000/ingest/uplink")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args()
    cfg = load_config(args.config if os.path.exists(args.config) else
                      os.path.join(ROOT, "config", "demo.example.toml"))
    sep = "&" if "?" in args.url else "?"
    url = f"{args.url}{sep}source=test" + (f"&token={cfg.ingest_token}" if cfg.ingest_token else "")
    for i in range(args.rounds):
        for dev in cfg.devices:
            req = urllib.request.Request(url, data=json.dumps(payload(dev, i)).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                print(dev.name, "->", r.read().decode())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
