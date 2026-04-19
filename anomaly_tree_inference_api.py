"""
FastAPI inference service for the decision tree trained on ensemble anomaly labels.

Train + export bundle (reproduces the notebook pipeline at a high level):

    uv sync
    uv run python anomaly_tree_inference_api.py train --csv bank_transactions_data_2.csv

Serve:

    uv run uvicorn anomaly_tree_inference_api:app --host 0.0.0.0 --port 8000

Environment:

    ARTIFACT_PATH   Path to joblib bundle (default: artifacts/dt_inference_bundle.joblib)

See README_INFERENCE_API.md for run, curl, and pytest instructions.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("anomaly_tree_inference_api")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    artifact_path: Path = Field(
        default=Path("artifacts/dt_inference_bundle.joblib"),
        validation_alias="ARTIFACT_PATH",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _configure_logging() -> None:
    """Attach a handler once if nothing is configured (e.g. tests); respect LOG_LEVEL."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    if not logger.handlers and not logging.root.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(h)
        logger.propagate = False


# ---------------------------------------------------------------------------
# Pydantic — raw transaction (matches bank CSV semantics)
# ---------------------------------------------------------------------------


class TransactionPayload(BaseModel):
    """Single transaction as in the source CSV (before notebook feature engineering)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    transaction_amount: float = Field(..., description="TransactionAmount")
    transaction_type: str = Field(..., description="Credit or Debit")
    location: str = Field(..., description="City / location label")
    device_id: str = Field(..., description="DeviceID")
    merchant_id: str = Field(..., description="MerchantID")
    channel: str = Field(..., description="ATM | Online | Branch")
    customer_age: int = Field(..., ge=0, le=120, description="CustomerAge")
    customer_occupation: str = Field(..., description="CustomerOccupation")
    transaction_duration: int = Field(..., ge=0, description="TransactionDuration (seconds)")
    login_attempts: int = Field(..., ge=0, description="LoginAttempts")
    account_balance: float = Field(..., description="AccountBalance")

    transaction_date: str = Field(
        ...,
        description="ISO-like timestamp, e.g. 2023-04-11 16:29:14",
    )

    @field_validator("transaction_date")
    @classmethod
    def parse_ts(cls, v: str) -> str:
        pd.to_datetime(v, format="mixed")
        return v


class PredictRequest(BaseModel):
    transactions: list[TransactionPayload] = Field(..., min_length=1)


class PredictionItem(BaseModel):
    label: Literal[0, 1] = Field(..., description="0 = Normal, 1 = Anomaly (tree prediction)")
    proba_normal: float
    proba_anomaly: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]


# ---------------------------------------------------------------------------
# Bundle I/O + training (mirrors transaction_risk_profile notebook logic)
# ---------------------------------------------------------------------------

DROP_COLS = [
    "TransactionID",
    "AccountID",
    "TransactionDate",
    "PreviousTransactionDate",
    "IP Address",
    "Amount_Z_Score",
    "period",
]

TREE_DROP_FROM_MODEL = [
    "IF_Anomaly",
    "LOF_Anomaly",
    "MCD_Score",
    "MCD_Anomaly",
    "OCSVM_Anomaly",
    "IF_Score",
    "OCSVM_Score",
    "Ensemble_Anomaly",
]


class LOFWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=20, contamination=0.05, metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.model = None

    def fit(self, X, y=None):
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
        )
        self.labels_ = self.model.fit_predict(X)
        return self

    def predict(self, X):
        return self.labels_


def _lof_scorer(estimator, X, y=None):
    labels = estimator.fit(X).labels_
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    return -1.0


def _label_encode_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    out = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    non_numeric = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
    for col in non_numeric:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le
    return out, encoders


def _safe_le_transform(le: LabelEncoder, value: str) -> int:
    value = str(value)
    if value in getattr(le, "classes_", []):
        return int(le.transform([value])[0])
    # unseen category → map to mode of training classes (first class is arbitrary fallback)
    return int(le.transform([le.classes_[0]])[0])


def _rows_to_model_matrix(
    rows: list[TransactionPayload], bundle: dict[str, Any]
) -> pd.DataFrame:
    """Build the same feature columns the tree saw (label-encoded, unscaled)."""
    encoders: dict[str, LabelEncoder] = bundle["label_encoders"]
    avg_by_type: dict[str, float] = bundle["avg_amount_by_type"]
    device_counts: dict[str, int] = bundle["device_tx_count"]

    records = []
    for r in rows:
        ts = pd.to_datetime(r.transaction_date, format="mixed")
        avg_type = avg_by_type.get(r.transaction_type, np.mean(list(avg_by_type.values())))
        ratio = r.transaction_amount / avg_type if avg_type else 0.0
        dev_cnt = int(device_counts.get(r.device_id, 0))
        records.append(
            {
                "TransactionAmount": r.transaction_amount,
                "TransactionType": r.transaction_type,
                "Location": r.location,
                "DeviceID": r.device_id,
                "MerchantID": r.merchant_id,
                "Channel": r.channel,
                "CustomerAge": r.customer_age,
                "CustomerOccupation": r.customer_occupation,
                "TransactionDuration": r.transaction_duration,
                "LoginAttempts": r.login_attempts,
                "AccountBalance": r.account_balance,
                "Hour": int(ts.hour),
                "DayOfWeek": int(ts.dayofweek),
                "Month": int(ts.month),
                "Amount_to_AvgByType_Ratio": float(ratio),
                "DeviceTxCount": dev_cnt,
            }
        )

    raw_df = pd.DataFrame.from_records(records)
    for col in raw_df.columns:
        if col in encoders and col in raw_df.columns:
            raw_df[col] = [_safe_le_transform(encoders[col], str(v)) for v in raw_df[col]]
    return raw_df


def train_bundle_from_csv(csv_path: Path, out_path: Path, *, fast: bool = True) -> None:
    """Rebuild df_model, ensemble labels, decision tree, and save joblib bundle."""
    logger.info("train_bundle_from_csv csv=%s out=%s fast=%s", csv_path, out_path, fast)
    df = pd.read_csv(csv_path)
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], format="mixed")

    df_model = df.copy()
    df_model["Hour"] = df["TransactionDate"].dt.hour
    df_model["DayOfWeek"] = df["TransactionDate"].dt.dayofweek
    df_model["Month"] = df["TransactionDate"].dt.month

    avg_tx_amount_by_type = df_model.groupby("TransactionType")["TransactionAmount"].transform("mean")
    df_model["Amount_to_AvgByType_Ratio"] = df_model["TransactionAmount"] / avg_tx_amount_by_type
    device_tx_count = df_model.groupby("DeviceID").size().reset_index(name="DeviceTxCount")
    df_model = df_model.merge(device_tx_count, on="DeviceID", how="left")

    df_model.drop(columns=[c for c in DROP_COLS if c in df_model.columns], inplace=True)

    df_model_encoded, label_encoders = _label_encode_frame(df_model)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_model_encoded)
    df_scaled_df = pd.DataFrame(df_scaled, columns=df_model_encoded.columns)

    # --- MCD ---
    mcd = MinCovDet(support_fraction=0.75, random_state=42)
    mcd.fit(df_scaled_df)
    mahal_dist = mcd.mahalanobis(df_scaled_df)
    threshold = float(np.percentile(mahal_dist, 97.5))
    df_model_encoded["MCD_Anomaly"] = (mahal_dist > threshold).astype(int)
    df_model_encoded["MCD_Score"] = mahal_dist

    # --- Isolation Forest ---
    if fast:
        iso = IsolationForest(
            n_estimators=200, contamination=0.05, max_samples="auto", random_state=42
        )
        iso.fit(df_scaled_df)
        best_iso_forest = iso
    else:
        param_grid = {
            "n_estimators": [100, 200],
            "contamination": [0.01, 0.02, 0.05],
            "max_samples": [256, "auto"],
        }
        grid = GridSearchCV(
            IsolationForest(random_state=42),
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=0,
        )
        grid.fit(df_scaled_df)
        best_iso_forest = grid.best_estimator_

    if_scores = best_iso_forest.decision_function(df_scaled_df)
    if_pred = best_iso_forest.predict(df_scaled_df)
    df_model_encoded["IF_Score"] = if_scores
    df_model_encoded["IF_Anomaly"] = pd.Series(if_pred).map({1: 0, -1: 1}).astype(int)

    # --- LOF ---
    if fast:
        lof = LocalOutlierFactor(
            n_neighbors=20, contamination=0.05, metric="euclidean", novelty=False
        )
        lof_labels = lof.fit_predict(df_scaled_df)
        best_lof_labels = np.where(lof_labels == -1, 1, 0)
    else:
        scorer = make_scorer(_lof_scorer, greater_is_better=True)
        n = len(df_scaled_df)
        cv_single = [(np.arange(n), np.arange(n))]
        param_grid = {
            "n_neighbors": [10, 20, 30, 50],
            "contamination": [0.01, 0.02, 0.05, 0.1],
            "metric": ["euclidean", "manhattan"],
        }
        grid_lof = GridSearchCV(
            LOFWrapper(),
            param_grid=param_grid,
            scoring=scorer,
            cv=cv_single,
            verbose=0,
        )
        grid_lof.fit(df_scaled_df)
        best_lof = grid_lof.best_estimator_
        best_lof_labels = np.where(best_lof.labels_ == -1, 1, 0)

    df_model_encoded["LOF_Anomaly"] = best_lof_labels.astype(int)

    # --- One-Class SVM ---
    ocsvm = OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
    ocsvm_pred = ocsvm.fit_predict(df_scaled_df)
    df_model_encoded["OCSVM_Score"] = ocsvm.decision_function(df_scaled_df)
    df_model_encoded["OCSVM_Anomaly"] = pd.Series(ocsvm_pred).map({1: 0, -1: 1}).astype(int)

    df_model_encoded["Ensemble_Anomaly"] = (
        (
            df_model_encoded["IF_Anomaly"]
            + df_model_encoded["LOF_Anomaly"]
            + df_model_encoded["OCSVM_Anomaly"]
            + df_model_encoded["MCD_Anomaly"]
        )
        >= 2
    ).astype(int)

    X = df_model_encoded.drop(columns=[c for c in TREE_DROP_FROM_MODEL if c in df_model_encoded.columns])
    y = df_model_encoded["Ensemble_Anomaly"]

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X, y)

    avg_amount_by_type = (
        df.groupby("TransactionType")["TransactionAmount"].mean().astype(float).to_dict()
    )
    device_tx_count = df.groupby("DeviceID").size().astype(int).to_dict()

    bundle = {
        "decision_tree": dt,
        "feature_names": list(X.columns),
        "label_encoders": label_encoders,
        "avg_amount_by_type": avg_amount_by_type,
        "device_tx_count": device_tx_count,
        "meta": {
            "fast_train": fast,
            "n_rows": int(len(df)),
            "tree_depth": int(dt.get_depth()),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    logger.info(
        "saved bundle to %s (%s features, %s rows)",
        out_path,
        len(X.columns),
        len(df),
    )
    print(f"Saved bundle to {out_path} ({len(X.columns)} features, {len(df)} rows).")


@lru_cache
def _load_bundle_cached(path_str: str) -> dict[str, Any]:
    p = Path(path_str)
    if not p.is_file():
        logger.error("bundle path is not a file: %s", p)
        raise FileNotFoundError(f"Artifact not found: {p}")
    logger.info("loading joblib bundle from %s", p)
    bundle = joblib.load(p)
    n_feat = len(bundle.get("feature_names", []))
    logger.info("bundle loaded (%s feature columns)", n_feat)
    return bundle


def load_bundle() -> dict[str, Any]:
    return _load_bundle_cached(str(get_settings().artifact_path.resolve()))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

_bundle_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle_error
    _configure_logging()
    settings = get_settings()
    logger.info(
        "application startup artifact_path=%s log_level=%s",
        settings.artifact_path.resolve(),
        settings.log_level,
    )
    try:
        load_bundle()
        _bundle_error = None
        logger.info("artifact load succeeded")
    except Exception as e:  # noqa: BLE001
        _bundle_error = str(e)
        logger.exception("artifact load failed: %s", e)
    yield
    logger.info("application shutdown")


app = FastAPI(
    title="Anomaly decision tree inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each HTTP request with latency (paths only; no request body)."""
    path = request.url.path
    skip = path in ("/favicon.ico",)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.exception(
            "request failed %s %s after %.1fms",
            request.method,
            path,
            elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not skip:
        lvl = logging.DEBUG if path == "/health" else logging.INFO
        logger.log(
            lvl,
            "%s %s -> %s %.1fms",
            request.method,
            path,
            getattr(response, "status_code", "?"),
            elapsed_ms,
        )
    return response


@app.get("/health")
def health():
    ok = _bundle_error is None
    body = {"status": "ok" if ok else "degraded", "detail": _bundle_error}
    logger.debug("health check status=%s", body["status"])
    return body


@app.get("/v1/feature-names")
def feature_names():
    if _bundle_error:
        logger.warning("feature_names rejected: bundle error=%s", _bundle_error)
        raise HTTPException(503, _bundle_error)
    b = load_bundle()
    names = b["feature_names"]
    logger.debug("feature_names count=%s", len(names))
    return {"features": names}


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _bundle_error:
        logger.warning("predict rejected: bundle unavailable (%s)", _bundle_error)
        raise HTTPException(503, _bundle_error)
    t0 = time.perf_counter()
    n = len(req.transactions)
    logger.info("predict request batch_size=%s", n)
    bundle = load_bundle()
    tree: DecisionTreeClassifier = bundle["decision_tree"]
    expected = list(bundle["feature_names"])

    Xp = _rows_to_model_matrix(req.transactions, bundle)
    missing = [c for c in expected if c not in Xp.columns]
    if missing:
        logger.error("feature matrix missing columns: %s", missing)
        raise HTTPException(500, f"Internal feature build missing columns: {missing}")
    Xp = Xp[expected].astype(np.float64)

    pred = tree.predict(Xp)
    proba = tree.predict_proba(Xp)
    cls_list = [int(c) for c in tree.classes_]
    idx = {c: j for j, c in enumerate(cls_list)}
    if 0 not in idx or 1 not in idx:
        logger.error("tree classes invalid: %s", cls_list)
        raise HTTPException(
            500,
            f"Tree must be trained on binary classes 0 and 1; got classes_={cls_list}",
        )
    i0, i1 = idx[0], idx[1]
    out: list[PredictionItem] = []
    for i in range(len(pred)):
        p = proba[i]
        out.append(
            PredictionItem(
                label=int(pred[i]),
                proba_normal=float(p[i0]),
                proba_anomaly=float(p[i1]),
            )
        )
    n_anom = sum(1 for item in out if item.label == 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "predict done batch_size=%s predicted_anomalies=%s duration_ms=%.1f",
        n,
        n_anom,
        elapsed_ms,
    )
    return PredictResponse(predictions=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def train_main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train decision tree + save joblib bundle.")
    p.add_argument("--csv", type=Path, default=Path("bank_transactions_data_2.csv"))
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output joblib path (default: ARTIFACT_PATH or artifacts/dt_inference_bundle.joblib)",
    )
    p.add_argument(
        "--full-grid",
        action="store_true",
        help="Run slower IF/LOF grid search similar to the notebook.",
    )
    args = p.parse_args(argv)
    out = args.out or get_settings().artifact_path
    train_bundle_from_csv(args.csv, out, fast=not args.full_grid)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_main(sys.argv[2:])
    else:
        print(
            "Usage:\n"
            "  uv run python anomaly_tree_inference_api.py train [--csv PATH] [--out PATH] [--full-grid]\n"
            "  uv run uvicorn anomaly_tree_inference_api:app --host 0.0.0.0 --port 8000",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
