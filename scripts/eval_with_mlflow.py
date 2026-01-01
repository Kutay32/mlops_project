"""
Evaluate a model from MLflow Model Registry and log metrics back to MLflow.

Usage:
  MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/eval_with_mlflow.py \
      --registered-model-name "UltimateChurnModel" --stage "Staging" --data-url <csv>
"""

import os
import argparse
import warnings

import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

warnings.filterwarnings("ignore")


def load_test_data(data_path: str):
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.dropna(subset=['Churn'])
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registered-model-name', type=str, required=True)
    parser.add_argument('--stage', type=str, default='Staging', help='Model registry stage to load (Staging or Production)')
    parser.add_argument('--data-url', type=str, required=True)
    args = parser.parse_args()

    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            print(f"Using MLflow tracking at: {mlflow_uri}")
        except Exception as e:
            print(f"Could not set MLflow tracking URI: {e}")

    X, y = load_test_data(args.data_url)

    try:
        import mlflow
        model_uri = f"models:/{args.registered_model_name}/{args.stage}"
        print(f"Loading model from registry: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"Could not load model from registry: {e}")
        return

    # Convert X to numpy array if necessary
    try:
        X_vals = X.values
    except Exception:
        X_vals = X

    preds = model.predict(X_vals)

    # If the model returns a DataFrame with probabilities, try to extract
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] > 1:
            y_pred = (preds[:, 1] >= 0.5).astype(int)
            proba = preds[:, 1]
        else:
            y_pred = preds
            proba = None

        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
        }
        if proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y, proba))

        print(f"Evaluation metrics: {metrics}")

        # Log metrics to MLflow
        try:
            with mlflow.start_run(run_name=f"eval_{args.registered_model_name}_{args.stage}"):
                for k, v in metrics.items():
                    mlflow.log_metric(k, float(v))
                mlflow.set_tag('evaluated_model', args.registered_model_name)
                mlflow.set_tag('evaluated_stage', args.stage)
        except Exception as e:
            print(f"Could not log evaluation metrics: {e}")

    except Exception as e:
        print(f"Error computing metrics: {e}")


if __name__ == '__main__':
    main()
