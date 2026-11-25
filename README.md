# LoRaWAN × AI Demonstrator — "Ahle Wurst meets KI"

Portable demonstrator (University of Kassel, ComTec): real LoRaWAN sensors send measurements via a
**Milesight UG63** gateway and a router to a laptop. The laptop decodes them, runs **three lightweight
CPU models** (Logistic Regression, Random Forest, MLP) and shows the results side by side in a browser.
The models classify the curing phase of a sausage (*Umrötung → Reifung Phase 1 → Reifung Phase 2*).

Without a sausage in the case, the **virtual sausage** tab takes a real recorded day and lets you shift
moisture, conductivity and temperature with sliders. For any moment, **"How was this decided?"** walks through
uplinks → features → which signals mattered → what happened inside each model. UI in German and English.

```
LSE01 / LTC-FS / WS202 ──LoRa──▶ UG63 (embedded network server) ──HTTP POST──▶ router/Wi-Fi ──▶ laptop
                                                                                 │
             dashboard (browser) ◀── FastAPI ◀── window features ◀── decode ─────┘
                    ▲                               │
                    └── replay / simulation ────────┴──▶ LogReg · RandomForest · MLP (CPU)
```

## Layout

| Path | Content |
|---|---|
| `notebooks/` | `01_convert_raw_data` → `02_search_hyperparameters` → `03_train_and_evaluate`, plus `04_check_seasonal_confound`. Outputs stay in the notebooks |
| `ml_core/` | Shared library used by the notebooks and the demonstrator: `data` (paths, columns), `windowing` (window features), `pipeline` (`ModelPipeline`), `conversion`, `training`, `augmentation` |
| `demonstrator/` | Laptop app: gateway ingest, live state, replay, simulation, explanations, dashboard (`static/`) |
| `data/raw/` | Excel exports `series_1.xlsx` (annotated), `series_2.xlsx` (not annotated), via Git LFS |
| `data/converted/` | One CSV per sausage, written by notebook 01, via Git LFS |
| `data/models/` | Written by notebooks 02/03: `hyperparameters.json`, `final/`, `folds/`, `manifest.json`, `metrics.json` |
| `config/demo.example.toml` | Device list (DevEUI → sensor type, sausage slot) and server settings |
| `scripts/` | `setup.ps1`, `start_demo.ps1` / `.sh`, `send_test_uplink.py` (fake uplinks without hardware) |
| `docs/` | `PLAN.md` (decisions, requirement mapping, open items), `RUNBOOK.md` (gateway setup, demo, packing), assignment |
| `tests/` | `python -m pytest -q`, on synthetic data (see below) |

## Quick start (Windows laptop)

```powershell
git clone <gitlab-url> ; cd masterproject-lorawan-demonstrator
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, packages, git lfs pull, config copy
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1  # opens http://localhost:8000/?lang=de
```

Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && git lfs pull`, then `scripts/start_demo.sh`.

Two views: `http://localhost:8000/` is the **presentation view** (one source at a time, consensus phase, three models,
four measurements, hardly any text); `/details` is the full dashboard (devices, timeline, evaluation, explanations).
Both need trained models in `data/models/` (see below). Uplinks from the case are stored in `data/recordings/`
(see [RUNBOOK](docs/RUNBOOK.md) §4). The UG63 HTTP integration must point to
`http://<laptop-ip>:8000/ingest/uplink`; the start script prints the URL. Details: [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Train the models

```powershell
.\.venv\Scripts\pip install -r requirements-notebooks.txt
.\.venv\Scripts\jupyter lab notebooks
```

1. **01_convert_raw_data**: only needed if the Excel exports change.
2. **02_search_hyperparameters**: searches the three model families (leave-one-sausage-out, balanced accuracy) and
   saves the winners to `data/models/hyperparameters.json`. Tens of minutes on CPU; uses a GPU via RAPIDS if installed.
3. **03_train_and_evaluate**: evaluates, trains and exports the models the demonstrator loads, and checks them on the
   unannotated series 2 (phases must advance in time there too). Without a search result it uses
   the defaults in `ml_core/training.py`. Headless alternative: `python -m ml_core.training`.
4. **04_check_seasonal_confound**: compares feature sets and augmentations. Series 1 (the only annotated run) goes
   from spring to summer, so the absolute temperature rises with the phase, and models that use it can learn the
   season. The default feature set therefore uses the probe temperatures *relative to the air* (plus moisture and
   conductivity). Optional training-only augmentation (`ml_core/augmentation.py`): probe calibration, noise and lost
   uplinks, temperature level and swings; switched on with `AUGMENT` in notebook 03 or `--augment` on the command line.

5. **05_compare_windows**: how much history a prediction should see: longer windows (e.g. 30 uplinks) and context
   features (mean, spread and trend per day over the last 24 h / 7 days per signal, `context` in `WindowFeatures`).
   Window size, stride and context are settings (`DEFAULT_WINDOW_SIZE`, `DEFAULT_CONTEXT` in `ml_core/training.py`);
   the demonstrator reads them from the exported models.

Unattended (e.g. overnight), all notebooks one after another, outputs saved in the notebooks as in Jupyter:

```powershell
.\.venv\Scripts\python scripts\run_notebooks.py > run.log 2>&1        # 01 → 04; stops at the first error
.\.venv\Scripts\python scripts\run_notebooks.py --from 02 --set N_JOBS=4  # skip conversion, override a setting
```

The notebook files are saved after every cell, so an open notebook shows how far the run is.

Windowing, cross-validation folds and the feature-set comparison run in parallel worker processes (`N_JOBS` in the
notebooks, default all cores; environment variable `ML_N_JOBS` for the command line). Each worker holds its own copy of
the training windows, so lower `N_JOBS` if RAM runs short.

Train with **the same scikit-learn version as the demo laptop** (pinned to 1.9.x in `requirements.txt`).
Model files are not portable across versions; the dashboard warns on a mismatch.

## Tests

`python -m pytest -q` (the notebook test also needs `requirements-notebooks.txt`). The tests use **synthetic** data
with the real file layout (`tests/synthetic_data.py`), so they run without the LFS files. The synthetic data is easy
to separate. Its scores say nothing about the real task.
