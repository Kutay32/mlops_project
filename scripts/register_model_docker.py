"""
Script to register model to MLflow - runs inside Docker container
Compatible with MLflow 2.x
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import joblib
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

def register_model():
    """Register the churn model to MLflow Model Registry."""
    
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
    print(f"📡 Connecting to MLflow at: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    client = MlflowClient()
    
    # Set experiment
    experiment_name = "Churn_Prediction_Production"
    mlflow.set_experiment(experiment_name)
    print(f"📊 Experiment: {experiment_name}")
    
    # Load the model
    model_path = "/app/model_artifact.joblib"
    if not os.path.exists(model_path):
        print(f"⚠ Model not found at {model_path}")
        return
    
    print(f"🔍 Loading model from: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Model loaded: {type(model).__name__}")
    
    # Start MLflow run
    model_name = "ChurnPredictionModel"
    run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 Starting run: {run_name}")
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("source_file", model_path)
        mlflow.log_param("registration_date", datetime.now().isoformat())
        mlflow.log_param("model_type", type(model).__name__)
        
        # Log pipeline info if available
        if hasattr(model, 'steps'):
            steps = [step[0] for step in model.steps]
            mlflow.log_param("pipeline_steps", str(steps))
        
        # Log metrics
        mlflow.log_metric("registered", 1)
        
        # Create sample input for signature inference
        # Define the expected input schema for churn prediction
        sample_input = pd.DataFrame({
            'gender': ['Male'],
            'SeniorCitizen': [0],
            'Partner': ['Yes'],
            'Dependents': ['No'],
            'tenure': [12],
            'PhoneService': ['Yes'],
            'MultipleLines': ['No'],
            'InternetService': ['Fiber optic'],
            'OnlineSecurity': ['No'],
            'OnlineBackup': ['Yes'],
            'DeviceProtection': ['No'],
            'TechSupport': ['No'],
            'StreamingTV': ['Yes'],
            'StreamingMovies': ['Yes'],
            'Contract': ['Month-to-month'],
            'PaperlessBilling': ['Yes'],
            'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [70.35],
            'TotalCharges': [844.2]
        })

        # Infer signature from sample prediction
        try:
            sample_pred = model.predict(sample_input)
            signature = infer_signature(sample_input, sample_pred)
        except Exception as sig_err:
            print(f"⚠ Could not infer signature: {sig_err}")
            signature = None

        # Log the model with signature
        print("📦 Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=sample_input,
        )
        
        print(f"✓ Model logged successfully!")
        print(f"  Run ID: {run.info.run_id}")
        
        # Try to transition to Production
        try:
            # Get latest version
            versions = client.get_latest_versions(model_name)
            if versions:
                latest = versions[0]
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest.version,
                    stage="Production"
                )
                print(f"✓ Model version {latest.version} promoted to Production")
        except Exception as e:
            print(f"⚠ Could not promote to Production: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model registration complete!")
    print("=" * 60)
    print(f"\nModel URI: models:/{model_name}/Production")
    print(f"MLflow UI: {mlflow_uri}")


if __name__ == "__main__":
    register_model()
