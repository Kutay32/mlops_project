# 🚀 MLOps Churn Prediction Project

Complete MLOps implementation for Customer Churn Prediction with ML Design Patterns.

## 📋 Contents

- [Features](#features)
- [Architecture](#architecture)
- [ML Design Patterns](#ml-design-patterns)
- [Quick Start](#quick-start)
- [Services](#services)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Monitoring](#monitoring)

## ✨ Features

- **ML Pipeline**: Automated feature engineering and model training
- **Model Serving**: FastAPI-based REST API with fallback support
- **Model Tracking**: MLflow for experiment tracking and model registry
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **Workflow Orchestration**: Prefect DAG-based pipelines
- **Data Validation**: Great Expectations-style data quality checks
- **CI/CD**: GitHub Actions pipeline
- **Checkpoints**: Training resumption support
- **Fallback**: Multi-tier fallback prediction system

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MLOps Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  Data Layer  │────▶│   Training   │────▶│   Serving    │                │
│  │              │     │   Pipeline   │     │     API      │                │
│  │ • Validation │     │ • Prefect    │     │ • FastAPI    │                │
│  │ • GE Checks  │     │ • MLflow     │     │ • Fallback   │                │
│  └──────────────┘     │ • Checkpoint │     └──────────────┘                │
│                       └──────────────┘           │                          │
│                                                  ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Monitoring Layer                               │  │
│  │   Prometheus ──────▶ Grafana ──────▶ Drift Detection                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        CI/CD Pipeline (GitHub Actions)                │  │
│  │   Code Quality ──▶ Unit Tests ──▶ Build ──▶ Deploy                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ ML Design Patterns Implemented

### Data Representation Patterns
| Pattern | Implementation | File |
|---------|---------------|------|
| 🔢 Hashed Feature | HashingEncoder | `src/data/feature_engineering.py` |
| 📊 Embeddings | EmbeddingEncoder | `src/data/feature_engineering.py` |
| ✖️ Feature Cross | FeatureCrosser | `src/data/feature_engineering.py` |

### Problem Representation Patterns
| Pattern | Implementation | File |
|---------|---------------|------|
| ⚖️ Rebalancing | SMOTE, ADASYN, Undersample | `src/models/base_model.py` |
| 🔄 Reframing | RegressionToClassification | `src/models/reframing.py` |

### Model Training Patterns
| Pattern | Implementation | File |
|---------|---------------|------|
| 🎯 Ensemble | Voting, Stacking, Bagging | `src/models/ensemble.py` |
| 💾 Checkpoints | CheckpointManager | `src/training/checkpoints.py` |
| ⏪ Fallback | FallbackChain | `src/serving/fallback.py` |

### Continuous Model Evaluation (CME)
| Pattern | Implementation | File |
|---------|---------------|------|
| 📈 Drift Detection | DriftDetector | `src/monitoring/production_monitor.py` |
| 📉 Performance Monitoring | PerformanceMonitor | `src/monitoring/production_monitor.py` |

## 🚀 Quick Start

### Requirements

- Docker Desktop
- Docker Compose v2.0+
- Python 3.9+

### 1. Start All Services

```bash
# Start API, MLflow, Prometheus, and Grafana
docker-compose up -d

# Follow logs
docker-compose logs -f
```

### 2. Run Model Training

```bash
# Run training job
docker-compose --profile training up training-job

# Or run Prefect workflow
cd prefect_flows
python training_flow.py
```

### MLflow Tracking & Model Registry

This repository includes an MLflow server in `docker-compose.yml` for experiment tracking and the Model Registry.

1. Start the MLflow server (or the whole stack):

```bash
# Start only MLflow server
docker-compose up -d mlflow-server

# Or start all services
docker-compose up -d
```

2. Run training with MLflow tracking and optional registry registration:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train_with_mlflow.py \
  --data-url "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv" \
  --registered-model-name "UltimateChurnModel"
```

3. Evaluate a registered model from the registry and log evaluation metrics:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/eval_with_mlflow.py \
  --registered-model-name "UltimateChurnModel" --stage "Staging" \
  --data-url "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
```

4. Promote a model version to Production (optionally archive previous Production versions):

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/promote_model.py \
  --registered-model-name "UltimateChurnModel" --version 1 --target-stage Production --archive-existing
```

5. MLflow UI: http://localhost:5000

> Quick demo: run `scripts/run_local_mlflow_demo.sh` to start MLflow, train a model, evaluate it, and promote the latest staging model to Production (uses Docker Compose and the example dataset).

---

### 3. Check Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## 🔧 Services

| Service | Port | Description |
|---------|------|-------------|
| `api-service` | 8000 | Prediction API with fallback |
| `mlflow-server` | 5000 | Model tracking and registry |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Visualization |
| `training-job` | - | Model training (batch) |

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Make Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
      "TotalCharges": 844.2
    }
  }'
```

### Fallback Statistics

```bash
curl http://localhost:8000/fallback-stats
```

### Model Info

```bash
curl http://localhost:8000/model-info
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_feature_engineering.py -v
```

## 📊 Data Validation

```python
from great_expectations import validate_telco_dataset
import pandas as pd

df = pd.read_csv("your_data.csv")
report = validate_telco_dataset(df)

if report.overall_success:
    print("✅ Validation passed!")
else:
    print("❌ Validation failed!")
    print(report.to_json())
```

## 🔄 Fallback Hierarchy

```
┌─────────────────────────────────────────────────┐
│                Prediction Request                │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│ Level 1: Primary ML Model (XGBoost/RF)          │
│          ↓ Error or Low Confidence              │
├─────────────────────────────────────────────────┤
│ Level 2: Secondary Model (Logistic Regression)  │
│          ↓ Error or Low Confidence              │
├─────────────────────────────────────────────────┤
│ Level 3: Business Rules Engine                  │
│          ↓ Error                                │
├─────────────────────────────────────────────────┤
│ Level 4: Historical Base Rate (27% churn)       │
└─────────────────────────────────────────────────┘
```

## 🛠️ Development

### Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn src.serving.api:app --reload
```

### Docker Build

```bash
# Build only API image
docker-compose build api-service

# Build all images
docker-compose build
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Also remove volumes
docker-compose down -v
```

## 📊 Monitoring

### Prometheus Metrics

API exposes metrics at `/metrics`:

- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration
- `http_requests_in_progress`: Active requests

### Grafana Dashboard

1. Go to http://localhost:3000
2. Login with admin/admin123
3. Prometheus datasource is auto-configured
4. Create or import dashboards

## 📁 Project Structure

```
mlops_project/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
├── docker/
│   ├── Dockerfile.serving         # API container
│   ├── Dockerfile.training        # Training container
│   └── Dockerfile.register        # Model registration container
├── great_expectations/
│   ├── __init__.py
│   └── data_validation.py         # Data quality validation
├── prefect_flows/
│   └── training_flow.py           # Prefect DAG workflow
├── src/
│   ├── data/
│   │   └── feature_engineering.py # High-cardinality patterns
│   ├── evaluation/
│   │   └── benchmarking.py        # Model benchmarking
│   ├── models/
│   │   ├── base_model.py          # Rebalancing patterns
│   │   ├── ensemble.py            # Ensemble patterns
│   │   └── reframing.py           # Problem reframing
│   ├── monitoring/
│   │   └── production_monitor.py  # Drift detection & CME
│   ├── serving/
│   │   ├── api.py                 # FastAPI serving
│   │   └── fallback.py            # Algorithmic fallback
│   └── training/
│       └── checkpoints.py         # Checkpoint pattern
├── tests/
│   ├── unit/
│   │   ├── test_feature_engineering.py
│   │   ├── test_models.py
│   │   └── test_monitoring.py
│   └── integration/
│       └── test_api.py
├── configs/
│   └── prometheus.yml             # Prometheus configuration
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

## 🔒 Security Notes

- Change Grafana password in production
- Use HTTPS
- Manage environment variables with secret manager

## 📝 License

MIT License
