"""HTTP tests for `anomaly_tree_inference_api` using dummy transaction payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLE = _ROOT / "artifacts" / "dt_inference_bundle.joblib"


pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import anomaly_tree_inference_api as api  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    if not _BUNDLE.is_file():
        pytest.skip(f"Missing bundle: {_BUNDLE} — export it from transaction_risk_profile.ipynb")
    api._load_bundle_cached.cache_clear()
    api.get_settings.cache_clear()
    return TestClient(api.app)


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["detail"] is None


def test_feature_names(client: TestClient) -> None:
    r = client.get("/v1/feature-names")
    assert r.status_code == 200
    names = r.json()["features"]
    assert isinstance(names, list)
    assert len(names) >= 1


def test_predict_dummy_single(client: TestClient) -> None:
    """One synthetic row shaped like the bank CSV (similar to TX000001)."""
    payload = {
        "transactions": [
            {
                "transaction_amount": 14.09,
                "transaction_type": "Debit",
                "location": "San Diego",
                "device_id": "D000380",
                "merchant_id": "M015",
                "channel": "ATM",
                "customer_age": 70,
                "customer_occupation": "Doctor",
                "transaction_duration": 81,
                "login_attempts": 1,
                "account_balance": 5112.21,
                "transaction_date": "2023-04-11 16:29:14",
            }
        ]
    }
    r = client.post("/v1/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    p0 = data["predictions"][0]
    assert p0["label"] in (0, 1)
    assert 0.0 <= p0["proba_normal"] <= 1.0
    assert 0.0 <= p0["proba_anomaly"] <= 1.0
    assert abs(p0["proba_normal"] + p0["proba_anomaly"] - 1.0) < 1e-5


def test_predict_dummy_batch(client: TestClient) -> None:
    payload = {
        "transactions": [
            {
                "transaction_amount": 100.0,
                "transaction_type": "Credit",
                "location": "Houston",
                "device_id": "D000051",
                "merchant_id": "M052",
                "channel": "ATM",
                "customer_age": 45,
                "customer_occupation": "Engineer",
                "transaction_duration": 120,
                "login_attempts": 2,
                "account_balance": 2500.0,
                "transaction_date": "2023-06-27 16:44:19",
            },
            {
                "transaction_amount": 5000.0,
                "transaction_type": "Debit",
                "location": "Chicago",
                "device_id": "D000264",
                "merchant_id": "M015",
                "channel": "Branch",
                "customer_age": 32,
                "customer_occupation": "Student",
                "transaction_duration": 300,
                "login_attempts": 5,
                "account_balance": 150.0,
                "transaction_date": "2023-12-01 10:00:00",
            },
        ]
    }
    r = client.post("/v1/predict", json=payload)
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 2
