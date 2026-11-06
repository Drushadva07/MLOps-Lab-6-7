
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_prediction():
    data = {
        "area_sqft": 1000,
        "bedrooms": 2,
        "age_years": 10,
        "location_score": 7
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_invalid_prediction():
    data = {
        "area_sqft": 1000,
        "bedrooms": 2,
        "age_years": 10,
        "location_score": 15
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422
