"""Entry point: python -m demonstrator --config config/demo.toml"""
from __future__ import annotations

import argparse
import logging
import os
import socket

import uvicorn

from .app import create_app
from .config import ROOT, load_config


def local_ips() -> list[str]:
    ips = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127."))


def main():
    ap = argparse.ArgumentParser(description="LoRaWAN AI demonstrator")
    default_cfg = os.path.join(ROOT, "config", "demo.toml")
    ap.add_argument("--config", default=default_cfg if os.path.exists(default_cfg) else None)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    host, port = args.host or cfg.host, args.port or cfg.port
    print(f"\n  Dashboard:        http://localhost:{port}/")
    for ip in local_ips():
        print(f"  Gateway HTTP URL: http://{ip}:{port}/ingest/uplink" + (f"?token={cfg.ingest_token}" if cfg.ingest_token else ""))
    print(f"  Config:           {cfg.path or '(none – no devices configured)'}\n")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
