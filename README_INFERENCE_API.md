# Decision tree anomaly inference API

This folder includes a small **FastAPI** service (`anomaly_tree_inference_api.py`) that loads a **joblib bundle** (decision tree + label encoders + lookup tables) and exposes **`POST /v1/predict`** for raw transaction–shaped JSON.

The bundle is meant to be produced from **`transaction_risk_profile.ipynb`** (export cell after `dt.fit`), so you normally **do not** run the optional `train` CLI unless you want a standalone retrain.

---

## Prerequisites

- Python **3.11+**
- A bundle file at **`artifacts/dt_inference_bundle.joblib`** (default path)

### Create the bundle from the notebook

1. Open `transaction_risk_profile.ipynb`.
2. Run the notebook through the **decision tree** cell (`dt.fit(X, y)` and rules/plot).
3. Run the **export** cell right after it. It writes `artifacts/dt_inference_bundle.joblib`.

### Install dependencies

**Option A — UV (recommended)**

```bash
cd c:\Users\Admin\Desktop\devoteam_test
uv sync
uv sync --extra dev
```

**Option B — pip**

```bash
pip install fastapi uvicorn pydantic pydantic-settings numpy pandas scikit-learn joblib
pip install httpx pytest   # for tests
```

---

## Run the API server

From the project directory (so the default `artifacts/` path resolves):

**Windows (PowerShell)**

```powershell
cd c:\Users\Admin\Desktop\devoteam_test
$env:ARTIFACT_PATH = "$PWD\artifacts\dt_inference_bundle.joblib"
uv run uvicorn anomaly_tree_inference_api:app --host 127.0.0.1 --port 8000
```

If you omit `ARTIFACT_PATH`, the app uses **`artifacts/dt_inference_bundle.joblib`** relative to the current working directory.

### Logging

- **`LOG_LEVEL`**: `DEBUG`, `INFO` (default), `WARNING`, or `ERROR` — controls the `anomaly_tree_inference_api` logger.
- On startup the app logs the resolved **`ARTIFACT_PATH`** and whether the bundle loaded.
- Each HTTP request logs **method, path, status code, and duration** (`/health` is logged at **DEBUG** to reduce noise).
- **`POST /v1/predict`** logs **batch size**, count of **predicted anomalies**, and **duration** (no raw transaction payloads).

To reduce duplicate lines with Uvicorn’s access log, you can disable its access logger:

`uvicorn anomaly_tree_inference_api:app --no-access-log`

**Generic**

```bash
export ARTIFACT_PATH=/path/to/dt_inference_bundle.joblib   # Linux/macOS
uvicorn anomaly_tree_inference_api:app --host 0.0.0.0 --port 8000
```

Open interactive docs: **http://127.0.0.1:8000/docs**

### Smoke test with curl

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/feature-names
curl -s -X POST http://127.0.0.1:8000/v1/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"transaction_amount\":14.09,\"transaction_type\":\"Debit\",\"location\":\"San Diego\",\"device_id\":\"D000380\",\"merchant_id\":\"M015\",\"channel\":\"ATM\",\"customer_age\":70,\"customer_occupation\":\"Doctor\",\"transaction_duration\":81,\"login_attempts\":1,\"account_balance\":5112.21,\"transaction_date\":\"2023-04-11 16:29:14\"}]}"
```

On PowerShell, `Invoke-RestMethod` is often easier than escaping JSON in `curl`.

---

## Request body shape (`POST /v1/predict`)

Field names are **snake_case** (Pydantic model `TransactionPayload`):

| Field | Example |
|--------|---------|
| `transaction_amount` | `14.09` |
| `transaction_type` | `Debit` |
| `location` | `San Diego` |
| `device_id` | `D000380` |
| `merchant_id` | `M015` |
| `channel` | `ATM` |
| `customer_age` | `70` |
| `customer_occupation` | `Doctor` |
| `transaction_duration` | `81` |
| `login_attempts` | `1` |
| `account_balance` | `5112.21` |
| `transaction_date` | `2023-04-11 16:29:14` |

Response: `predictions[]` with `label` (0 normal, 1 anomaly), `proba_normal`, `proba_anomaly`.

---

## Automated tests (dummy data)

Requires **`httpx`** (for Starlette/FastAPI `TestClient`) and **`pytest`**, and the **joblib bundle** present under `artifacts/`.

```bash
cd c:\Users\Admin\Desktop\devoteam_test
uv sync --extra dev
uv pip install pytest
uv run pytest tests/test_anomaly_inference_api.py -v
```

If the bundle is missing, tests **skip** with a message telling you to run the notebook export cell.

---

## Optional: train bundle without the notebook

If you do not have a notebook export, you can rebuild a compatible bundle from the CSV (not identical hyperparameter search to your last notebook run unless you use `--full-grid`):

```bash
uv run python anomaly_tree_inference_api.py train --csv bank_transactions_data_2.csv --out artifacts/dt_inference_bundle.joblib
```

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `503` / health `degraded` | `ARTIFACT_PATH` points to a real file; run export or `train` |
| Import errors | `uv sync` or install deps from `pyproject.toml` |
| Wrong working directory | Start uvicorn from the repo root, or set `ARTIFACT_PATH` to an absolute path |
