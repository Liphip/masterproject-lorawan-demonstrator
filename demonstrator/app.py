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
    """Build the FastAPI app around one set of shared state objects.

    Routes:
        POST /ingest/uplink      uplinks from the UG63 HTTP integration (or scripts/send_test_uplink.py)
        GET  /api/state          everything the dashboards poll: models, live devices, replay
        GET  /api/metrics        metrics.json of the trained models
        GET  /api/recordings     recorded runs available for replay / simulation
        POST /api/replay/...     start, stop, pause a replay
        GET  /api/explain        explanation of the prediction on screen
        POST /api/simulate       virtual sausage (what-if on a recorded window)
        POST /api/ambient        air temperature around the case
        GET  /  and  /details    presentation view and full dashboard
    """
    models = ModelRegistry(cfg.models_dir)
    live = LiveState(cfg, models)
    replay = ReplayEngine(models, cfg.replay_dirs)
    simulator = Simulator(models, cfg.replay_dirs)
    started = time.time()

    @asynccontextmanager
    async def lifespan(_app):
        """Stop the replay thread when the server shuts down."""
        yield
        replay.stop()

    app = FastAPI(title="LoRaWAN AI Demonstrator", docs_url="/api/docs", redoc_url=None,
                  lifespan=lifespan)
    app.state.live, app.state.replay, app.state.models = live, replay, models

    # ── gateway ingest (UG63 HTTP integration posts here) ───────────────────
    @app.post("/ingest/uplink")
    async def ingest(request: Request, token: str = "", source: str = "live"):
        """Accept one uplink or a list of uplinks (JSON). ?source=test marks test uplinks (not recorded);
        ?token=... is required if ingest_token is set in the config.
        """
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
        """Run blocking work (decoding, model inference) in a thread so the event loop stays responsive."""
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(fn, *args)

    # ── API ─────────────────────────────────────────────────────────────────
    @app.get("/api/state")
    def state():
        """Snapshot polled by both dashboards about every 1.5 s."""
        return {"server_time": time.time(), "uptime_s": time.time() - started,
                "models": models.info(),
                "ingest": {"http_path": "/ingest/uplink", "token_required": bool(cfg.ingest_token),
                           "config_file": cfg.path or None,
                           "configured_devices": len(cfg.devices)},
                "live": live.snapshot(), "replay": replay.snapshot()}

    @app.get("/api/metrics")
    def metrics():
        """metrics.json as written by training (404 if the models have not been trained yet)."""
        if not models.metrics:
            raise HTTPException(404, "metrics.json not found – run notebooks/03_train_and_evaluate.ipynb")
        return models.metrics

    @app.get("/api/recordings")
    def recordings():
        """Recorded runs (converted training data and live recordings) available for replay and simulation."""
        return replay.recordings()

    @app.post("/api/replay/start")
    async def replay_start(request: Request):
        """Start replaying a recording: {group, speed (data seconds per second), start_fraction 0..1}."""
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
        """Stop the replay (its history stays visible)."""
        replay.stop()
        return {"ok": True}

    @app.post("/api/replay/pause")
    async def replay_pause(request: Request):
        """Pause or resume the replay: {paused: true|false}."""
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

    @app.post("/api/ambient")
    async def set_ambient(request: Request):
        """Air temperature around the case (the case has no air sensor)."""
        body = await request.json()
        try:
            value = float(body["value"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "expected {\"value\": <°C>}")
        if not -30 <= value <= 50:
            raise HTTPException(400, "air temperature must be between -30 and 50 °C")
        await _run_sync(live.set_ambient, value)
        return {"ok": True, "value": value}

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
        """Liveness check for scripts: is the server up, are the models loaded."""
        return JSONResponse({"ok": True, "models_ok": models.ok})

    @app.get("/")
    def presentation():
        """Presentation view: one prediction, three models, few words."""
        return FileResponse(os.path.join(STATIC, "simple.html"))

    @app.get("/details")
    def details():
        """Full dashboard: devices, data path, timeline, evaluation, explanations."""
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
