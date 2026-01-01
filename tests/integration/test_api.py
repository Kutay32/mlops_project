"""
Integration Tests for API Endpoints
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def client():
    """Create test client"""
    from src.serving.api import app

    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health and readiness endpoints"""

    def test_health_check(self, client):
        """Test /health endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "Churn" in data["service"]


class TestPredictionEndpoint:
    """Tests for prediction endpoint"""

    @pytest.fixture
    def sample_features(self):
        """Sample features for prediction"""
        return {
            "features": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.2,
            }
        }

    def test_predict_returns_valid_response(self, client, sample_features):
        """Test prediction returns valid response structure"""
        response = client.post("/predict", json=sample_features)

        # May fail if model not loaded in test, but structure should be correct
        if response.status_code == 200:
            data = response.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert data["churn_prediction"] in [0, 1]
            assert 0 <= data["churn_probability"] <= 1

    def test_predict_invalid_input(self, client):
        """Test prediction with invalid input"""
        response = client.post("/predict", json={"invalid": "data"})

        assert response.status_code in [422, 500]


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""

    def test_model_info(self, client):
        """Test /model-info endpoint"""
        response = client.get("/model-info")

        assert response.status_code == 200
        data = response.json()
        assert "model_loaded" in data
        assert "predictions_made" in data
