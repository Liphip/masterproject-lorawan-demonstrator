# Project plan

Status as of 23 Nov 2026. Scope: [assignment requirements](ASSIGNMENT_REQUIREMENTS.md)
(A1–A11). The principle: **demonstrate the whole chain reliably**. This is not a general IoT platform.

## Architecture decisions

| Decision | Choice | Why / trade-off |
|---|---|---|
| Gateway → laptop transport | UG63 **embedded network server + HTTP POST** to the laptop | No broker, no cloud, works offline. MQTT stays optional (`[mqtt]` in the config) if the gateway is later set up that way. |
| Decoding | Payload codec **on the gateway** preferred; built-in fallback for LSE01 and WS202 | Vendor codecs are maintained for the firmware. The fallback decoders are written from the documented layouts and must be checked against real uplinks. LTC-FS has no fallback. |
| App | One Python process (FastAPI) + custom HTML/JS dashboard **instead of Grafana** (original idea) | The audience is companies seeing a showcase: the UI has to tell a story (data path → models → *why*). Grafana is excellent for time series, but per-prediction explanations, replay control and the pause-and-explain flow would need plugins or awkward workarounds, plus InfluxDB/Prometheus on site. Cost: the dashboard is our own code to maintain (~700 lines JS, no build step, no CDN, works offline). |
| Explaining predictions | "How was this decided?" overlay. (1) the uplinks in the window, (2) the 29 features with training range, (3) **one** occlusion test for all models (hide one sensor signal → change in confidence), (4) model-specific internals: LogReg weighted sum (exact), RF tree votes + Saabas split contributions, MLP neuron activations | (3) makes the models comparable, and (4) shows the three different principles, which is the point of A6/A7. Occlusion and Saabas are approximations: correlated signals share credit. This is stated in the UI. SHAP would be more rigorous but adds a dependency and seconds of latency per click. |
| Live vs. recorded | Two separate panels; replay is always labeled REPLAY in orange | Curing takes days (§3.1). Replay is where the model comparison is meaningful; live proves the data path. |
| Honest replay | Replaying a labeled sausage uses the **fold model trained without it** | Otherwise the demo would show memorised training data. The unlabeled series 2 (independent run) uses the final models. |
| No sausage in the case | Three views of one system: **live** (real sensors, proves the LoRaWAN path), **replay** (a real recorded run with held-out models), **virtual sausage** (a recorded window from any day, levels shifted by sliders, models re-predict instantly) | Considered: mixing live readings into a recorded run, so that a live uplink advances the "sausage". Rejected: live air readings mixed with sausage readings would be a third, unrealistic kind of data and hard to label honestly. The simulator keeps real uplink timing and noise and changes only levels, and is labelled SIMULATED everywhere. Limit: shifting a level does not create a realistic *history* (e.g. a sausage that dried slowly). It answers "what does the model do with these numbers?", not "what would this sausage really do?". |
| Live compatibility | Out-of-range check against training feature percentiles | Implements "only show live predictions where data is compatible": the dashboard warns instead of hiding. |
| Random Forest size | Demo preset `max_depth=10, n_estimators=100` (HPO winner was unbounded, 500 trees) | An unbounded forest on ~160k windows grows to hundreds of MB. Compare both with `--rf-preset hpo` and report the difference. |

## Requirement mapping

| ID | Delivered by | Status |
|---|---|---|
| A1 LoRaWAN system | UG63 config ([RUNBOOK](RUNBOOK.md) §2), `/ingest/uplink` | **Code done; hardware setup open** |
| A2 Readable measurements | Dashboard panel 1 (sensor, DevEUI, values with units, RSSI/SNR, age, uplink log) | Done |
| A3 ML on laptop | `demonstrator/inference.py`, scikit-learn on CPU | Done |
| A4 Visualisation | Data path strip, panels 1–3 | Done; to refine after a dry run |
| A5 Transport case | Physical | **Open: assembly, cable/power plan, photo docs** |
| A6 ≥3 model families | LogReg, RF, MLP | Done |
| A7 Side-by-side + trade-offs | Model cards, agreement, phase timeline, evaluation table (quality, latency, size, training time), explanation overlay | Done |
| A8 CPU-only | Latency per prediction measured and shown; single-threaded inference | Done; **check on the demo laptop** |
| A9 Replay, labeled | `demonstrator/replay.py`; live uplinks recorded in the same schema → replayable | Done |
| A10 Repeatable operation | `scripts/setup.ps1`, `start_demo.ps1`, RUNBOOK | Done; **test with a second person** |
| A11 Limitations | Panel 4, single-run note in metrics, out-of-range and "seen in training" warnings | Done |

## Changes to existing code (and why)

- `ml_app/preprocessing/windowing.py`
  - **Slope feature fixed.** It used the first/last row of the window, which almost always belongs to another
    sensor, so every `*_slope` feature was NaN and the imputer dropped it (see the notebook warnings). It now uses
    the first/last valid reading of the sensor. **Consequence: the notebook's HPO results were obtained without slope
    features. Re-run the HPO if time permits.**
  - Timestamps are normalised to ns before the int64 cast. pandas ≥ 3 can parse ISO strings at µs resolution,
    which would silently scale all interval features by 1000 between environments.
  - Optional `output_columns` fixes the feature schema, so that windows where a sensor did not report (common live) still
    match the trained models.
- `ml_app/pipeline/pipeline.py`: `created_at` default was evaluated once at import time.
- Notebook cells 12–13 are superseded by `train_models.py`. `Pipeline.fit` with the windowing step windows X but
  not y, so those cells cannot run. Logistic Regression is wrapped in `OneVsRestClassifier`, because scikit-learn ≥ 1.8
  dropped implicit one-vs-rest for `liblinear` (same model as before).

## Open items (in order)

0. ~~Sensors in the case vs. training sensors.~~ Resolved 24 Nov: the case holds **WS202, LSE01, LTC-FS**, the same types as
   the training data. Live values still come from air rather than a sausage. That is handled by the virtual sausage and the out-of-range warning.
1. **Run `train_models.py` on the real data** and look at `metrics.json`. Check whether the slope fix changed scores.
2. **Collect sensor data:** DevEUI/AppEUI/AppKey for every sensor; check that the sensors are not still joined to another network
   server (TTN, company LNS).
3. **UG63 setup** following RUNBOOK §2. Capture one real uplink JSON per sensor type (`data/recordings/raw_*.jsonl`) and
   compare the decoded values with the device app/manual. Add a test with the captured payloads.
4. **Dry run** on the demo laptop: startup time, latency, battery, offline behaviour.
5. Case assembly and packing list (A5). Document exact sensor and power-supply models.
6. **Check for shortcut learning with the explain view.** If hiding *uplink timing* changes predictions more than hiding the
   sausage measurements (seen on the synthetic data), the models may have learned the sensors' send rhythm instead of the
   curing. Then retrain without the `timestamp_*` features and compare scores. Either outcome is a good point for the presentation.
7. Optional: longer windows (e.g. 6–12 h) are likely more meaningful for a multi-day process than 10 uplinks. This is a
   small change via `--window-size`, but the HPO would need to be re-run. Worth one experiment for the evaluation chapter.

## Known risks

- **Single labeled run.** Scores are per sausage within one run and are probably optimistic. Say so in the presentation.
- **Short windows.** 10 uplinks ≈ minutes. The models mostly classify current absolute levels (moisture, conductivity,
  temperature), not the dynamics of the run.
- **Gateway clock.** The laptop's receive time is used as the timestamp, because an offline gateway may have no NTP.
- **Laptop IP.** The gateway URL contains the laptop IP. Use a DHCP reservation or a static IP on the demo router.
