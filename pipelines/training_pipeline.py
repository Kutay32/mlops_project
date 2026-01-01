"""
MLOps Training Pipeline with MLflow Integration
"""

import os
import sys
import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

# Add src to path for imports
sys.path.insert(0, "/app")


def wait_for_mlflow(uri: str, max_retries: int = 30, delay: int = 2):
    """Wait for MLflow server to be ready."""
    import requests

    for i in range(max_retries):
        try:
            response = requests.get(f"{uri}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ MLflow server is ready at {uri}")
                return True
        except Exception as e:
            print(f"Waiting for MLflow server... ({i+1}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError(f"MLflow server not available at {uri}")


def load_data():
    """Load and prepare the Telco Churn dataset."""
    print("📥 Loading dataset...")
    url = "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Basic cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Handle target
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
        if df["Churn"].dtype == "object":
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def train_model(df: pd.DataFrame):
    """Train the ML model and log to MLflow."""
    import joblib
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    print("🔧 Building pipeline...")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Define feature columns
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    # Train 3 models: RandomForest, XGBoost, LightGBM
    models = {
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
    }

    best_model = None
    best_score = 0
    best_model_name = None

    for model_name, classifier in models.items():
        print(f"\n🚀 Training {model_name}...")

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", classifier)]
        )

        with mlflow.start_run(
            run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", X_train.shape[0])
            mlflow.log_param("test_size", 0.2)

            # Train
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predict
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("training_time_seconds", training_time)

            # Infer model signature from training data
            signature = infer_signature(X_train, y_pred)

            # Log model with registration and signature
            registered_name = f"ChurnModel_{model_name}"
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=registered_name,
                signature=signature,
                input_example=X_train.head(3),
            )

            print(f"  ✓ Accuracy: {accuracy:.4f}")
            print(f"  ✓ F1 Score: {f1:.4f}")
            print(f"  ✓ ROC AUC: {roc_auc:.4f}")
            print(f"  ✓ Training time: {training_time:.2f}s")
            print(f"  ✓ Registered as: {registered_name}")

            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model = pipeline
                best_model_name = model_name

    print(f"\n🏆 Best model: {best_model_name} with F1 Score: {best_score:.4f}")

    # Register best model to Production stage
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Get the best model's registered name
        best_registered_name = f"ChurnModel_{best_model_name}"

        # Get latest version and promote to Production
        versions = client.get_latest_versions(best_registered_name)
        if versions:
            latest = versions[0]
            client.transition_model_version_stage(
                name=best_registered_name, version=latest.version, stage="Production"
            )
            print(
                f"✓ Model {best_registered_name} v{latest.version} promoted to Production"
            )
    except Exception as e:
        print(f"⚠ Could not promote to Production: {e}")

    # Save best model as artifact
    model_path = "/app/model_artifact.joblib"
    joblib.dump(best_model, model_path)
    print(f"✓ Best model saved to {model_path}")

    return best_model, best_score


def main():
    """Main training pipeline execution."""
    print("=" * 60)
    print("🚀 MLOps Training Pipeline Started")
    print("=" * 60)

    # Get MLflow URI from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"\n📡 MLflow Tracking URI: {mlflow_uri}")

    # Wait for MLflow server
    try:
        wait_for_mlflow(mlflow_uri)
    except RuntimeError as e:
        print(f"⚠️ Warning: {e}")
        print("Continuing without MLflow tracking...")
        mlflow.set_tracking_uri("file:///app/mlruns")
    else:
        mlflow.set_tracking_uri(mlflow_uri)

    # Set experiment
    experiment_name = "Churn_Prediction_Pipeline"
    mlflow.set_experiment(experiment_name)
    print(f"📊 Experiment: {experiment_name}")

    # Load data
    df = load_data()

    # Train model
    model, score = train_model(df)

    print("\n" + "=" * 60)
    print("✅ Training Pipeline Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
