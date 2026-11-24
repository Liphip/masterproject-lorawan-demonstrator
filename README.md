# LoRaWAN × AI Demonstrator — "Ahle Wurst meets KI"

Portable demonstrator (University of Kassel, ComTec): real LoRaWAN sensors send measurements via a
**Milesight UG63** gateway and a router to a laptop. The laptop decodes them, runs **three lightweight
CPU models** (Logistic Regression, Random Forest, MLP) and shows the results side by side in a browser.
The models classify the curing phase of a sausage (*Umrötung → Reifung 1 → Reifung 2*).
Without a sausage in the case, the **virtual sausage** tab takes a real recorded day and lets you shift moisture, conductivity and temperature with sliders. For any moment, **"How was this decided?"** opens a step-by-step explanation: uplinks → features → which signals mattered →
what happened inside each model.

```
LSE01 / LTC-FS / WS202 ──LoRa──▶ UG63 (embedded network server) ──HTTP POST──▶ router/Wi-Fi ──▶ laptop
                                                                                 │
             dashboard (browser) ◀── FastAPI ◀── windowing + features ◀── decode ┘
                    ▲                               │
                    └── replay of recorded runs ────┴──▶ LogReg · RandomForest · MLP (CPU)
```

| Folder | Content |
|---|---|
| `ml_app/` | Shared preprocessing (windowing, feature extraction), `CustomPipeline`, data conventions (`data.py`) |
| `data-preperation/` | Notebook: raw Excel → `data/training/converted/*.csv` |
| `training/01_create_pipelines.ipynb` | Hyperparameter search (HPO) |
| `training/train_models.py` | Evaluation (leave-one-sausage-out) + model export → `data/models/` |
| `demonstrator/` | Laptop app: gateway ingest, live state, replay, inference, dashboard |
| `config/demo.example.toml` | Device list (DevEUI → sensor type, sausage slot) and server settings |
| `scripts/` | `setup.ps1`, `start_demo.ps1` / `.sh`, `send_test_uplink.py` |
| `docs/PLAN.md` | Project plan, requirement mapping (A1–A11), open items |
| `docs/RUNBOOK.md` | Gateway setup, demo procedure, packing, troubleshooting |

## Quick start (Windows laptop)

```powershell
git clone <gitlab-url> ; cd masterproject-lorawan-demonstrator
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # venv, deps, git lfs pull, config copy
.\.venv\Scripts\python.exe training\train_models.py              # ~minutes on CPU; writes data\models\
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1  # opens http://localhost:8000/
```

Linux: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && git lfs pull`,
then `.venv/bin/python training/train_models.py` and `scripts/start_demo.sh`.

**Without hardware:** start the demonstrator, then `python scripts/send_test_uplink.py --rounds 12`
pushes fake uplinks (marked *test*, never recorded) through the same HTTP path as the gateway.

**Gateway:** point the UG63 HTTP uplink integration at `http://<laptop-ip>:8000/ingest/uplink`.
The start script prints the URL. See [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Language

The dashboard is available in **German and English** (DE/EN switch top right, also in the explanation overlay).
It starts in the browser's language; `http://localhost:8000/?lang=de` forces German. The choice is remembered per browser.
All texts are in `demonstrator/static/i18n.js`. Technical error messages from the server stay in English.

## Train the models

`training/train_models.py` reuses the HPO winners from the notebook. It produces:

- `data/models/final/` — models fitted on all labeled sausages (used for live data and the unlabeled series 2)
- `data/models/folds/<sausage>/` — models fitted **without** that sausage; replaying a labeled sausage uses these,
  so the replayed predictions are real held-out predictions
- `metrics.json` — per-fold accuracy, balanced accuracy, macro-F1, confusion matrices, CPU latency, model size,
  pairwise model agreement
- `manifest.json` — window settings, feature schema, training feature ranges (for the out-of-range warning), library versions

Options: `--rf-preset hpo` (the notebook's unbounded Random Forest; very large files), `--no-fold-models`,
`--window-size/--stride`. **Train with the same scikit-learn version that runs the demo.** The dashboard warns on a mismatch.

## Tests

`python -m pytest -q`. The tests run on a **synthetic** dataset with the real schema (`tests/synthetic_data.py`),
because the real CSVs are Git LFS objects. The synthetic data is trivially separable. Its scores say nothing about the real task.
