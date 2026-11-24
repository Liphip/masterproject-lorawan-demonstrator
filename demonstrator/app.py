"""FastAPI app: gateway ingest endpoint, JSON API and the static dashboard."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .explain import explain
from .inference import ModelRegistry
from .live import LiveState
from .replay import ReplayEngine
from .simulate import Simulator

log = logging.getLogger(__name__)
STATIC = os.path.join(os.path.dirname(__file__), "static")


def create_app(cfg: Config) -> FastAPI:
    models = ModelRegistry(cfg.models_dir)
    live = LiveState(cfg, models)
    replay = ReplayEngine(models, cfg.replay_dirs)
    simulator = Simulator(models, cfg.replay_dirs)
    started = time.time()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        replay.stop()

    app = FastAPI(title="LoRaWAN AI Demonstrator", docs_url="/api/docs", redoc_url=None,
                  lifespan=lifespan)
    app.state.live, app.state.replay, app.state.models = live, replay, models

    # ── gateway ingest (UG63 HTTP integration posts here) ───────────────────
    @app.post("/ingest/uplink")
    async def ingest(request: Request, token: str = "", source: str = "live"):
        if cfg.ingest_token and token != cfg.ingest_token:
            raise HTTPException(401, "bad token")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "body is not JSON")
        source = "test" if source == "test" else "live"
        msgs = body if isinstance(body, list) else [body]
        results = [await _run_sync(live.handle_uplink, m, source) for m in msgs if isinstance(m, dict)]
        return {"results": results}

    async def _run_sync(fn, *args):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(fn, *args)

    # ── API ─────────────────────────────────────────────────────────────────
    @app.get("/api/state")
    def state():
        return {"server_time": time.time(), "uptime_s": time.time() - started,
                "models": models.info(),
                "ingest": {"http_path": "/ingest/uplink", "token_required": bool(cfg.ingest_token),
                           "config_file": cfg.path or None,
                           "configured_devices": len(cfg.devices)},
                "live": live.snapshot(), "replay": replay.snapshot()}

    @app.get("/api/metrics")
    def metrics():
        if not models.metrics:
            raise HTTPException(404, "metrics.json not found – run notebooks/03_train_and_evaluate.ipynb")
        return models.metrics

    @app.get("/api/recordings")
    def recordings():
        return replay.recordings()

    @app.post("/api/replay/start")
    async def replay_start(request: Request):
        body = await request.json()
        if not models.ok:
            raise HTTPException(409, models.error or "models not loaded")
        try:
            await _run_sync(replay.start, body["group"], float(body.get("speed", 1800)),
                            float(body.get("start_fraction", 0)))
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))
        return {"ok": True}

    @app.post("/api/replay/stop")
    def replay_stop():
        replay.stop()
        return {"ok": True}

    @app.post("/api/replay/pause")
    async def replay_pause(request: Request):
        body = await request.json()
        replay.set_paused(bool(body.get("paused", True)))
        return {"ok": True}

    @app.get("/api/explain")
    def explain_endpoint(source: str = "replay", slot: str = ""):
        """Explain the prediction currently on screen (replay) or the latest live window."""
        if not models.ok:
            raise HTTPException(409, models.error or "models not loaded")
        if source == "simulate":
            if simulator.last is None:
                raise HTTPException(404, "no simulation yet")
            rows, group, res = simulator.last
            out = explain(models, rows, group)
            out.update(source="simulate", group=group, time=res["time"], truth=res["truth"],
                       adjust=res["adjust"])
            return out
        if source == "replay":
            rows, group, latest = replay.current_window()
            if rows is None:
                raise HTTPException(404, "no replay prediction yet")
            out = explain(models, rows, group)
            out.update(source="replay", group=group,
                       time=latest.get("time") if latest else None,
                       truth=latest.get("truth") if latest else None)
        else:
            rows = live.window_rows(slot)
            out = explain(models, rows, None)
            out.update(source="live", group=slot, truth=None,
                       time=rows["timestamp"].iloc[-1].isoformat() if len(rows) else None)
        return out

    @app.post("/api/simulate")
    async def simulate(request: Request):
        """What-if on a recorded window: {group, position 0..1, adjust{...}}."""
        if not models.ok:
            raise HTTPException(409, models.error or "models not loaded")
        body = await request.json()
        try:
            return await _run_sync(simulator.run, body["group"], float(body.get("position", 0.5)),
                                   body.get("adjust") or {})
        except (KeyError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"ok": True, "models_ok": models.ok})

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
