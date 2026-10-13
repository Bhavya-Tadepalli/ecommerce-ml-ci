
import os
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Ecommerce High-Value Purchase Prediction API" in response.data

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_predict_without_json(client):
    response = client.post("/predict", json=None)
    assert response.status_code == 400 or response.status_code == 500

def test_model_artifact_exists():
    assert os.path.exists("artifacts/ecommerce_pipeline.joblib")
