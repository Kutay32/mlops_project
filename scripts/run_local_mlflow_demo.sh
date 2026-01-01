#!/usr/bin/env bash
set -euo pipefail

# Simple local demo that starts MLflow server and runs training, evaluation, and promotion
DATA_URL="https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
REGISTERED_MODEL_NAME="UltimateChurnModel"
MLFLOW_URI="http://localhost:5000"

echo "Starting MLflow server via docker-compose..."
docker-compose up -d mlflow-server

# Wait for MLflow health
echo "Waiting for MLflow to become healthy..."
for i in {1..30}; do
  if curl -sSf ${MLFLOW_URI}/health >/dev/null 2>&1; then
    echo "MLflow is up"
    break
  fi
  echo "Waiting..."
  sleep 2
done

export MLFLOW_TRACKING_URI=${MLFLOW_URI}

echo "Running training script with MLflow tracking..."
python scripts/train_with_mlflow.py --data-url "${DATA_URL}" --registered-model-name "${REGISTERED_MODEL_NAME}"

# Evaluate staging model
echo "Evaluating model in Staging..."
python scripts/eval_with_mlflow.py --registered-model-name "${REGISTERED_MODEL_NAME}" --stage "Staging" --data-url "${DATA_URL}"

# Promote latest to Production and archive existing
echo "Promoting latest staging model to Production..."
python scripts/promote_model.py --registered-model-name "${REGISTERED_MODEL_NAME}" --target-stage Production --archive-existing

echo "Demo complete. Visit MLflow UI at ${MLFLOW_URI}"