# Project plan

Status as of 25 Nov 2026 (evening). Scope: [assignment requirements](ASSIGNMENT.md)
(A1–A11). The principle: **demonstrate the whole chain reliably**. This is not a general IoT platform.

## Architecture decisions

| Decision | Choice | Why / trade-off |
|---|---|---|
| Gateway → laptop transport | UG63 **embedded network server + HTTP POST** to the laptop | No broker, no cloud, works offline. (An MQTT path was prototyped and removed again; it is in the git history if HTTP ever fails.) |
| Decoding | Payload codec **on the gateway** preferred; built-in fallback for LSE01 and WS202 | Vendor codecs are maintained for the firmware. The fallback decoders are written from the documented layouts and must be checked against real uplinks. LTC-FS has no fallback. |
| App | One Python process (FastAPI) + custom HTML/JS dashboard **instead of Grafana** (original idea) | The audience is companies seeing a showcase: the UI has to tell a story (data path → models → *why*). Grafana is excellent for time series, but per-prediction explanations, replay control and the pause-and-explain flow would need plugins or awkward workarounds, plus InfluxDB/Prometheus on site. Cost: the dashboard is our own code to maintain (~1,100 lines JS incl. texts, no build step, no CDN, works offline). Two views: a presentation view for talks (`/`) and the full dashboard (`/details`). |
| Explaining predictions | "How was this decided?" overlay. (1) the uplinks in the window, (2) the window features with their training range (24 with the default feature set), (3) **one** occlusion test for all models (hide one sensor signal → change in confidence), (4) model-specific internals: LogReg weighted sum (exact), RF tree votes + Saabas split contributions, MLP neuron activations | (3) makes the models comparable, and (4) shows the three different principles, which is the point of A6/A7. Occlusion and Saabas are approximations: correlated signals share credit. This is stated in the UI. SHAP would be more rigorous but adds a dependency and seconds of latency per click. |
| Honest replay | Replaying a labeled sausage uses the **fold model trained without it** | Otherwise the demo would show memorised training data. The unlabeled series 2 (independent run) uses the final models. |
| No sausage in the case | Three views of one system: **live** (real sensors, proves the LoRaWAN path), **replay** (a real recorded run with held-out models), **virtual sausage** (a recorded window from any day, levels shifted by sliders, models re-predict instantly) | Considered: mixing live readings into a recorded run, so that a live uplink advances the "sausage". Rejected: live air readings mixed with sausage readings would be a third, unrealistic kind of data and hard to label honestly. The simulator keeps real uplink timing and noise and changes only levels, and is labelled SIMULATED everywhere. Limit: shifting a level does not create a realistic *history* (e.g. a sausage that dried slowly). It answers "what does the model do with these numbers?", not "what would this sausage really do?". |
| Seasonal confound | Default feature set **`temperature_difference`**: moisture, conductivity and both probe temperatures *minus the air temperature* (no absolute temperature, no uplink timing). The unannotated series 2 (summer → spring) is used as a label-free check: predicted phases must advance in time (Spearman ρ of time vs. phase, shown in notebook 03 and the dashboard). Notebook 04 compares the feature sets | Series 1 (30.03.–02.08.2023, attic without climate control) has temperature rising with the phase, so a model can learn "summer" instead of "ripe" (already noted in the 2024 seminar paper). The series-1 score cannot reveal this; series 2 can. The difference removes the seasonal level and keeps the sausage's own behaviour (a moist sausage stays cooler than the air). The air temperature comes from the LSN50-v2-S31 in the raw data, aligned to every uplink in the conversion; the case has no air sensor, so live data uses a value set in the config/UI, while replay and simulation use the recorded air temperature. Series 2 is not used for training: it has no labels, its timing is unknown (probably slower in the cold), and labels from the model itself would inherit its bias. |
| Augmentation | Optional, training data only, chosen by name (`AUGMENTATIONS` in `ml_core/augmentation.py`): **calibration** (per probe and day: moisture ± 0.5/1/2 points, conductivity ± ~10/25/50 %), **robust** (measurement noise, 15 % lost uplinks), **temperature** (level offset, scaled/mirrored day-night swings). Calibration and robust change the raw rows and re-compute the windows (once per recording); temperature works on the window features in closed form. Default since 25 Nov 2026: `calibration_large` (`DEFAULT_AUGMENTATION`), see results | The scores differ more between sausages (0.58–0.81) than between models, so sausage/probe differences are the bigger weakness than the season. Calibration targets that; success = a better *worst sausage*, not only a better mean. Robust aims at live operation (irregular uplinks, noise); the held-out score can only show what it costs. Magnitudes are judgement calls from the data, not probe specifications. Too strong a moisture shift erases the signal (the phases differ by a few points), hence three strengths. Time stretching ("a cold run ripens slower") is not possible with windows of a few minutes. |
| Fourth model family | **Gradient boosting** (`hgb`, scikit-learn `HistGradientBoostingClassifier`) as a candidate: searched (02) and compared (04) by default, exported only if `MODELS` in notebook 03 includes it. The dashboards show any number of models; its explanation shows the phase probabilities as trees are added | Usually the strongest model for tabular features, CPU-only, small, no new dependency. It does not address the main limit (windows of minutes, see open items); a sequence model (1D-CNN/GRU over days) would, but needs more context per prediction, PyTorch and more data than 5 sausages to be trustworthy. |
| Case recordings | Every live uplink raw (`raw_<day>.jsonl`, before decoding) plus decoded rows per slot (`live_<day>_<slot>.csv`), flushed to disk; the last 6 h are reloaded at start | Nothing from the case is lost if a decoder is wrong or the laptop restarts; recordings are replayable like training data. |
| Live compatibility | Out-of-range check against training feature percentiles | Implements "only show live predictions where data is compatible": the dashboard warns instead of hiding. |
| Random Forest size | Demo cap `max_depth ≤ 10, n_estimators ≤ 100` (`SIZE_LIMITS`; latest search: depth 10, 200 trees) | An unbounded forest on ~100k windows grows to hundreds of MB; 100 trees score within 0.01 of 200 in the search. Compare with `LIMIT_SIZE = False` in notebook 03 if needed. |

## Requirement mapping

| ID | Delivered by | Status |
|---|---|---|
| A1 LoRaWAN system | UG63 config ([RUNBOOK](RUNBOOK.md) §2), `/ingest/uplink` | **Code done; hardware setup open** |
| A2 Readable measurements | Dashboard panel 1 (sensor, DevEUI, values with units, RSSI/SNR, age, uplink log) | Done |
| A3 ML on laptop | `demonstrator/inference.py`, scikit-learn on CPU | Done |
| A4 Visualisation | Presentation view (`/`: consensus phase, three models, four values, three sources) + full dashboard (`/details`: data path strip, panels 1–3) | Done; to refine after a dry run |
| A5 Transport case | Physical | **Open: assembly, cable/power plan, photo docs** |
| A6 ≥3 model families | LogReg, RF, MLP | Done |
| A7 Side-by-side + trade-offs | Model cards, agreement, phase timeline, evaluation table (quality, series-2 phase order, latency, size), explanation overlay; notebooks 03/04 for the full numbers | Done |
| A8 CPU-only | Latency per prediction measured and shown; single-threaded inference | Done; **check on the demo laptop** |
| A9 Replay, labeled | `demonstrator/replay.py`; live uplinks recorded in the same schema → replayable | Done |
| A10 Repeatable operation | `scripts/setup.ps1`, `start_demo.ps1`, RUNBOOK | Done; **test with a second person** |
| A11 Limitations | Panel 4, single-run note in metrics, out-of-range and "seen in training" warnings | Done |

## Changes to the original code (and why)

- **Feature bugs fixed** (`ml_core/windowing.py`, formerly `ml_app/preprocessing/windowing.py`):
  - The slope used the window's first/last row, which almost always belongs to another sensor, so every `*_slope`
    feature was NaN and the imputer dropped it. It now uses the first/last valid reading of the sensor.
    The first hyperparameter search ran without slope features; it has been re-run since.
  - Timestamps are normalised to ns before the int64 cast. pandas ≥ 3 can parse at µs resolution, which would scale
    all interval features by 1000 between environments.
  - `output_columns` fixes the feature schema, so windows in which a sensor did not report still match the models.
- **Conversion bug fixed:** an operator-precedence error in the conversion notebook would have written empty CSVs for
  every unannotated file (series 2) on the next run. The committed CSVs predate the bug and are fine.
- **Dead code removed:** `feature_extraction.py` (never called; the window features are computed vectorised in
  `windowing.py`), `training_only/wrappers.py` (imported but unused by the search), the label translation in the
  conversion notebook (never applied), MQTT ingest and the display-only sensor type.
- **Restructured:** logic lives in `ml_core/` (tested), the four notebooks only call it and keep their output;
  `scripts/run_notebooks.py` runs them unattended.
  The search writes `data/models/hyperparameters.json`, which training reads, so the winners are no longer copied by hand.
  `ModelPipeline.fit` raises a clear error, because a plain `fit` cannot work with a windowing step (X is windowed, y is not).
  This was also why the original final notebook cells failed.
- **Renamed:** `ml_app` → `ml_core`, `CustomPipeline` → `ModelPipeline`, `WindowingMethod` → `WindowFeatures`,
  pipeline step `inputer` → `imputer`, `data/training/{raw,converted}` → `data/{raw,converted}`,
  `raw.xlsx`/`raw2.xlsx` → `series_1.xlsx`/`series_2.xlsx`. Model files trained before this change must be retrained.
- **Logistic Regression** is always one-vs-rest (`OneVsRestClassifier`): scikit-learn ≥ 1.8 dropped the implicit
  one-vs-rest of `liblinear`. The search uses `l1_ratio` (0 = L2, 1 = L1) instead of the deprecated `penalty`.

## Results so far (real data, 25 Nov 2026)

- **Air temperature** found in both exports (series 1 "Sausage Temperature Sensor 1", series 2 "Temperature2",
  2–26 °C, coverage ≥ 99.9 %). Probe − air is small (−1.5 … +1.5 °C) and rises from about −1 °C to 0 °C while the
  sausage dries, in both runs.
- **Seasonal confound confirmed** (notebook 04, series 2 cut at February 2024): phase order ρ `original` 0.54–0.65,
  all restricted feature sets 0.76–0.84. Series-1 balanced accuracy: `original` 0.71–0.78, `temperature_difference`
  0.68–0.71. Default stays `temperature_difference` (as good as `no_temperature`, keeps temperature in the demo).
- **Series 2** advances Umrötung (August) → Phase 1 (mid/late September) → Phase 2 (from October, flickering with
  Phase 1). From early February 2024 LSE01 moisture jumps (≈ 7 → 22 %) and all models fall back: no log exists,
  probes moved or re-inserted is plausible, so the check stops at `PROGRESSION_UNTIL` (2024-02-01).
- **Temperature augmentation** changes nothing (the swings of the difference are small).
- **Per sausage** balanced accuracy ranges 0.58–0.81; sausage 1 ≈ 0.60 for every model and feature set.
- **Logistic Regression** search: best C = 100 at the grid edge, liblinear does not converge there (warnings in the
  log); the neighbouring settings score within 0.005.
- **Calibration augmentation (third run):** the random forest gains with strength: balanced accuracy 0.71 → 0.76 /
  0.78 / 0.79 (small / medium / large), series-2 ρ 0.80 → 0.87, backwards share 0.08 → 0.00; sausage 4 0.58 → 0.74,
  sausage 3 0.81 → 0.92. Logistic regression and MLP change by ≤ 0.01. `robust` costs 0.00–0.03. **Sausage 1 stays at
  0.60 for every model, feature set and augmentation**: not a probe-level difference that augmentation can model;
  more likely it was ahead of/behind the batch (labels) or a sensor problem.

## Open items (in order)

1. **Overnight run with gradient boosting:** notebook 02 (searches `hgb`; the other results are kept if only
   `MODELS = ["hgb"]`), then 03 and 04 (adds `calibration_xl`). Export `hgb` (`MODELS` in notebook 03) only if it beats
   the random forest on mean *and* worst sausage *and* series-2 order.
2. **Sausage 1:** look at its raw curves against the others (notebook 01 `SAMPLES`): offset, drift, a jump? If its
   probe misbehaved, say so in the evaluation; if not, it shows the limit of batch-level labels.
3. **Longer context** (most promising for accuracy): add day-scale features per window (e.g. moisture/conductivity
   trend over the last 24 h and 7 days; not the time since the start, which would be the schedule again). This
   needs the history of each sausage, which the replay and the live path both have. A sequence model would be the
   next step after that, not before.
4. **Collect sensor data:** DevEUI/AppEUI/AppKey for every sensor; check that the sensors are not still joined to another
   network server (TTN, company LNS).
5. **UG63 setup** following RUNBOOK §2. Capture one real uplink JSON per sensor type (`data/recordings/raw_*.jsonl`) and
   compare the decoded values with the device app/manual. Add a test with the captured payloads.
6. **Dry run** on the demo laptop: startup time, latency, battery, offline behaviour, both views; set the room's air
   temperature.
7. Case assembly and packing list (A5). Document exact sensor and power-supply models.
8. Optional: widen the Logistic Regression grid beyond C = 100 (or raise `max_iter`) if it stays relevant.

Done since the last plan: calibration/robust augmentation, faster replay (up to 1 week/s) and whole-run view, readable
recording names, persistent case recordings, model-count-independent UI, uplink-timing shortcut (the default feature set no longer contains `timestamp_*`; it only
matters for `original`), slope bug re-search, parallel training, presentation view, unattended notebook runs.

## Known risks

- **Single labeled run.** Scores are per sausage within one run and are probably optimistic. Say so in the presentation.
- **Labels are batch-level judgements.** The butcher recorded when the *whole batch* (hundreds of sausages without
  sensors) reached each phase, by his criteria and experience. So the labels do describe ripening, but: (a) every
  sensor sausage gets the batch's transition dates, although single sausages may be ahead or behind; (b) a transition
  is a date on which it was noticed, so windows near a boundary are uncertain. Both cap the achievable score and
  explain most Phase 1 ↔ Phase 2 confusion. (c) The seasonal confound remains: in series 1, ripening and rising
  temperature happen together, so the labels cannot tell the models which of the two to learn.
- **Augmentation magnitudes are judgement calls**, not probe specifications; notebook 04 decides whether they help.
- **Air temperature in the case is a setting**, not a measurement. Live predictions depend on it being roughly right.
- **Short windows.** 10 uplinks ≈ minutes. The models mostly classify current absolute levels (moisture, conductivity,
  temperature), not the dynamics of the run.
- **Gateway clock.** The laptop's receive time is used as the timestamp, because an offline gateway may have no NTP.
- **Laptop IP.** The gateway URL contains the laptop IP. Use a DHCP reservation or a static IP on the demo router.
