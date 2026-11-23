#!/usr/bin/env bash
# Linux/macOS equivalent of start_demo.ps1. First time:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && git lfs pull
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || { echo "Create the venv first (see header)"; exit 1; }
[ -f data/models/manifest.json ] || echo "WARNING: no trained models in data/models (see README)"
CFG=config/demo.toml; [ -f "$CFG" ] || CFG=config/demo.example.toml
( sleep 3; xdg-open http://localhost:8000/ >/dev/null 2>&1 || true ) &
exec .venv/bin/python -m demonstrator --config "$CFG"
