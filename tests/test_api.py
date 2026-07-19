import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(trained_artifact, monkeypatch):
    monkeypatch.setenv("DROPOUT_MODEL_PATH", str(trained_artifact))

    import api.main as api_main

    api_main._predictor = None  # reset any cached predictor from a previous test
    return TestClient(api_main.app)


def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_probability(client, cleaned_df):
    sample = cleaned_df.drop(columns=["target_binary"]).iloc[0].to_dict()
    resp = client.post("/predict", json={"features": sample})

    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["dropout_probability"] <= 1.0
    assert isinstance(body["at_risk"], bool)


def test_predict_missing_model_returns_503(monkeypatch):
    monkeypatch.setenv("DROPOUT_MODEL_PATH", "nonexistent/model.joblib")

    import api.main as api_main

    api_main._predictor = None
    client = TestClient(api_main.app)

    resp = client.post("/predict", json={"features": {"age_at_enrollment": 20}})
    assert resp.status_code == 503
