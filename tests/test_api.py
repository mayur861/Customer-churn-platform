from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["features"] == 10


def test_customer_prediction():
    payload = {
        "SeniorCitizen": 0,
        "Dependents": "No",
        "tenure": 5,
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
    assert 0 <= data["churn_probability_percent"] <= 100

    assert data["risk_level"] in [
        "Low", "Medium", "High", "Critical"
    ]

    assert isinstance(data["risk_drivers"], list)
    assert isinstance(data["protective_factors"], list)
    assert isinstance(data["retention_recommendations"], list)


def test_invalid_input():
    payload = {
        "SeniorCitizen": 0,
        "Dependents": "No"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422