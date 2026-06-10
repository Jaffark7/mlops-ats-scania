from fastapi.testclient import TestClient

from aps_failure.api import app


client = TestClient(app)


def test_health_endpoint_reports_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["feature_count"] > 100


def test_predict_endpoint_returns_binary_prediction_and_probability():
    sample_input = {
        "features": {
            "aa_000": 0.0,
            "ab_000": 1.0,
            "ac_000": 0.5,
        }
    }

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    body = response.json()
    assert body["failure_prediction"] in {0, 1}
    assert 0.0 <= body["failure_probability"] <= 1.0
