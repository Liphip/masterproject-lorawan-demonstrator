"""FastAPI app: gateway ingest endpoint, JSON API and the static dashboard."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .inference import ModelRegistry
from .live import LiveState
from .replay import ReplayEngine

log = logging.getLogger(__name__)
STATIC = os.path.join(os.path.dirname(__file__), "static")


class MqttIngest:
    """Optional: subscribe to a broker the gateway publishes to."""

    def __init__(self, cfg: Config, live: LiveState):
        self.cfg, self.live = cfg, live
        self.status = {"enabled": cfg.mqtt_enabled, "connected": False, "error": None}
        self.client = None

    def start(self):
        if not self.cfg.mqtt_enabled:
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            self.status["error"] = "paho-mqtt not installed"
            return
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="lorawan-ai-demonstrator")
        if self.cfg.mqtt_username:
            c.username_pw_set(self.cfg.mqtt_username, self.cfg.mqtt_password)

        def on_connect(client, userdata, flags, rc, props=None):
            self.status.update(connected=(rc == 0), error=None if rc == 0 else f"connect rc={rc}")
            client.subscribe(self.cfg.mqtt_topic)

        def on_disconnect(client, userdata, flags, rc, props=None):
            self.status["connected"] = False

        def on_message(client, userdata, msg):
            try:
                self.live.handle_uplink(json.loads(msg.payload), source="live")
            except Exception as e:  # noqa: BLE001
                self.live._error(f"MQTT message on {msg.topic} not processed: {e!r}")

        c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
        try:
            c.connect_async(self.cfg.mqtt_host, self.cfg.mqtt_port)
            c.loop_start()
            self.client = c
        except Exception as e:  # noqa: BLE001
            self.status["error"] = repr(e)

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def create_app(cfg: Config) -> FastAPI:
    models = ModelRegistry(cfg.models_dir)
    live = LiveState(cfg, models)
    replay = ReplayEngine(models, cfg.replay_dirs)
    mqtt_ingest = MqttIngest(cfg, live)
    started = time.time()

    @asynccontextmanager
    async def lifespan(_app):
        mqtt_ingest.start()
        yield
        replay.stop()
        mqtt_ingest.stop()

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
                           "mqtt": mqtt_ingest.status, "config_file": cfg.path or None,
                           "configured_devices": len(cfg.devices)},
                "live": live.snapshot(), "replay": replay.snapshot()}

    @app.get("/api/metrics")
    def metrics():
        if not models.metrics:
            raise HTTPException(404, "metrics.json not found – run training/train_models.py")
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

    @app.post("/api/models/reload")
    def models_reload():
        replay.stop()
        models.reload()
        return models.info()

    @app.get("/healthz")
    def healthz():
        return JSONResponse({"ok": True, "models_ok": models.ok})

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
