# Project plan

Status as of 24 Nov 2026. Scope: [assignment requirements](ASSIGNMENT.md)
(A1–A11). The principle: **demonstrate the whole chain reliably**. This is not a general IoT platform.

## Architecture decisions

| Decision | Choice | Why / trade-off |
|---|---|---|
| Gateway → laptop transport | UG63 **embedded network server + HTTP POST** to the laptop | No broker, no cloud, works offline. (An MQTT path was prototyped and removed again; it is in the git history if HTTP ever fails.) |
| Decoding | Payload codec **on the gateway** preferred; built-in fallback for LSE01 and WS202 | Vendor codecs are maintained for the firmware. The fallback decoders are written from the documented layouts and must be checked against real uplinks. LTC-FS has no fallback. |
| App | One Python process (FastAPI) + custom HTML/JS dashboard **instead of Grafana** (original idea) | The audience is companies seeing a showcase: the UI has to tell a story (data path → models → *why*). Grafana is excellent for time series, but per-prediction explanations, replay control and the pause-and-explain flow would need plugins or awkward workarounds, plus InfluxDB/Prometheus on site. Cost: the dashboard is our own code to maintain (~700 lines JS, no build step, no CDN, works offline). |
| Explaining predictions | "How was this decided?" overlay. (1) the uplinks in the window, (2) the 29 features with training range, (3) **one** occlusion test for all models (hide one sensor signal → change in confidence), (4) model-specific internals: LogReg weighted sum (exact), RF tree votes + Saabas split contributions, MLP neuron activations | (3) makes the models comparable, and (4) shows the three different principles, which is the point of A6/A7. Occlusion and Saabas are approximations: correlated signals share credit. This is stated in the UI. SHAP would be more rigorous but adds a dependency and seconds of latency per click. |
| Honest replay | Replaying a labeled sausage uses the **fold model trained without it** | Otherwise the demo would show memorised training data. The unlabeled series 2 (independent run) uses the final models. |
| No sausage in the case | Three views of one system: **live** (real sensors, proves the LoRaWAN path), **replay** (a real recorded run with held-out models), **virtual sausage** (a recorded window from any day, levels shifted by sliders, models re-predict instantly) | Considered: mixing live readings into a recorded run, so that a live uplink advances the "sausage". Rejected: live air readings mixed with sausage readings would be a third, unrealistic kind of data and hard to label honestly. The simulator keeps real uplink timing and noise and changes only levels, and is labelled SIMULATED everywhere. Limit: shifting a level does not create a realistic *history* (e.g. a sausage that dried slowly). It answers "what does the model do with these numbers?", not "what would this sausage really do?". |
| Seasonal confound | Default feature set **`temperature_difference`**: moisture, conductivity and both probe temperatures *minus the air temperature* (no absolute temperature, no uplink timing). Optional training-only augmentation of the temperature features (random offset, scaled and mirrored swings). The unannotated series 2 (summer → spring) is used as a label-free check: predicted phases must advance in time (Spearman ρ of time vs. phase, shown in notebook 03 and the dashboard). Notebook 04 compares the feature sets | Series 1 (30.03.–02.08.2023, attic without climate control) has temperature rising with the phase, so a model can learn "summer" instead of "ripe" (already noted in the 2024 seminar paper). The series-1 score cannot reveal this; series 2 can. The difference removes the seasonal level and keeps the sausage's own behaviour (a moist sausage stays cooler than the air). The air temperature comes from the LSN50-v2-S31 in the raw data, aligned to every uplink in the conversion; the case has no air sensor, so live data uses a value set in the config/UI, while replay and simulation use the recorded air temperature. Series 2 is not used for training: it has no labels, its timing is unknown (probably slower in the cold), and labels from the model itself would inherit its bias. |
| Live compatibility | Out-of-range check against training feature percentiles | Implements "only show live predictions where data is compatible": the dashboard warns instead of hiding. |
| Random Forest size | Demo preset `max_depth=10, n_estimators=100` (HPO winner was unbounded, 500 trees) | An unbounded forest on ~160k windows grows to hundreds of MB. Compare with `LIMIT_SIZE = False` in notebook 03 and report the difference. |

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

## Changes to the original code (and why)

- **Feature bugs fixed** (`ml_core/windowing.py`, formerly `ml_app/preprocessing/windowing.py`):
  - The slope used the window's first/last row, which almost always belongs to another sensor, so every `*_slope`
    feature was NaN and the imputer dropped it. It now uses the first/last valid reading of the sensor.
    **The first hyperparameter search ran without slope features; re-run notebook 02.**
  - Timestamps are normalised to ns before the int64 cast. pandas ≥ 3 can parse at µs resolution, which would scale
    all interval features by 1000 between environments.
  - `output_columns` fixes the feature schema, so windows in which a sensor did not report still match the models.
- **Conversion bug fixed:** an operator-precedence error in the conversion notebook would have written empty CSVs for
  every unannotated file (series 2) on the next run. The committed CSVs predate the bug and are fine.
- **Dead code removed:** `feature_extraction.py` (never called; the window features are computed vectorised in
  `windowing.py`), `training_only/wrappers.py` (imported but unused by the search), the label translation in the
  conversion notebook (never applied), MQTT ingest and the display-only sensor type.
- **Restructured:** logic lives in `ml_core/` (tested), the three notebooks only call it and keep their output.
  The search writes `data/models/hyperparameters.json`, which training reads, so the winners are no longer copied by hand.
  `ModelPipeline.fit` raises a clear error, because a plain `fit` cannot work with a windowing step (X is windowed, y is not).
  This was also why the original final notebook cells failed.
- **Renamed:** `ml_app` → `ml_core`, `CustomPipeline` → `ModelPipeline`, `WindowingMethod` → `WindowFeatures`,
  pipeline step `inputer` → `imputer`, `data/training/{raw,converted}` → `data/{raw,converted}`,
  `raw.xlsx`/`raw2.xlsx` → `series_1.xlsx`/`series_2.xlsx`. Model files trained before this change must be retrained.
- **Logistic Regression** is always one-vs-rest (`OneVsRestClassifier`): scikit-learn ≥ 1.8 dropped the implicit
  one-vs-rest of `liblinear`. The search uses `l1_ratio` (0 = L2, 1 = L1) instead of the deprecated `penalty`.

## Open items (in order)

1. **Run notebooks 02 and 03 on the real data** and look at the results. Compare with the first search (defaults in
   `ml_core/training.py`) to see whether the slope fix changed anything.
2. **Check the air temperature column** on the real data: notebook 01 reports which Excel column delivered it. Series 1:
   "Sausage Temperature Sensor 1" (per the seminar paper); series 2: *assumed* "Temperature2". Fix `AMBIENT_COLUMNS` in
   `ml_core/conversion.py` if wrong, then re-run notebook 01.
3. **Run notebook 04 on the real data** and choose the variant: good on series 1 *and* advancing on series 2. Set
   `DEFAULT_FEATURE_SET` / `AUGMENT` accordingly. On synthetic data the augmentation did not change anything; the real
   day/night swings may differ.
4. **Collect sensor data:** DevEUI/AppEUI/AppKey for every sensor; check that the sensors are not still joined to another network
   server (TTN, company LNS).
5. **UG63 setup** following RUNBOOK §2. Capture one real uplink JSON per sensor type (`data/recordings/raw_*.jsonl`) and
   compare the decoded values with the device app/manual. Add a test with the captured payloads.
6. **Dry run** on the demo laptop: startup time, latency, battery, offline behaviour.
7. Case assembly and packing list (A5). Document exact sensor and power-supply models.
8. **Check for shortcut learning with the explain view.** If hiding *uplink timing* changes predictions more than hiding the
   sausage measurements (seen on the synthetic data), the models may have learned the sensors' send rhythm instead of the
   curing. Then retrain without the `timestamp_*` features and compare scores. Either outcome is a good point for the presentation.
9. Optional: longer windows (e.g. 6–12 h) are likely more meaningful for a multi-day process than 10 uplinks. This is a
   small change (`WINDOW_SIZE` in notebooks 02 and 03), but the search has to be re-run. Worth one experiment for the evaluation chapter.

## Known risks

- **Single labeled run.** Scores are per sausage within one run and are probably optimistic. Say so in the presentation.
- **Labels are a schedule.** The butcher annotated series 1 by schedule, so every model learns "which phase of the
  schedule would this be under spring/summer conditions", not a measured ripeness. Restricting the inputs to the
  sausage's physical state (conductivity, moisture, temperature relative to the air) is what should let this carry over
  to colder runs, where the same state is reached later. No available score can prove that it is the true ripening.
- **Air temperature in the case is a setting**, not a measurement. Live predictions depend on it being roughly right.
- **Short windows.** 10 uplinks ≈ minutes. The models mostly classify current absolute levels (moisture, conductivity,
  temperature), not the dynamics of the run.
- **Gateway clock.** The laptop's receive time is used as the timestamp, because an offline gateway may have no NTP.
- **Laptop IP.** The gateway URL contains the laptop IP. Use a DHCP reservation or a static IP on the demo router.
