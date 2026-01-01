"""
Script to register existing model artifacts to MLflow Model Registry
"""
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def register_model_to_mlflow():
    """Register the trained churn model to MLflow."""
    
    # MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"📡 Connecting to MLflow at: {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    client = MlflowClient()
    
    # Set experiment
    experiment_name = "Churn_Prediction_Models"
    mlflow.set_experiment(experiment_name)
    print(f"📊 Experiment: {experiment_name}")
    
    # Find model files - try ultimate_churn_model first as it doesn't have custom classes
    model_files = [
        ("ultimate_churn_model.joblib", "UltimateChurnModel"),
    ]
    
    for model_path, model_name in model_files:
        if os.path.exists(model_path):
            print(f"\n🔍 Found: {model_path}")
            
            try:
                # Load the model
                model = joblib.load(model_path)
                print(f"✓ Model loaded successfully")
                print(f"  Model type: {type(model).__name__}")
                
                # Start MLflow run
                run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                with mlflow.start_run(run_name=run_name) as run:
                    # Log parameters
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("source_file", model_path)
                    mlflow.log_param("registration_date", datetime.now().isoformat())
                    
                    # Log model type info
                    model_type = type(model).__name__
                    mlflow.log_param("model_type", model_type)
                    
                    # If it's a pipeline, get the classifier info
                    if hasattr(model, 'steps'):
                        steps = [step[0] for step in model.steps]
                        mlflow.log_param("pipeline_steps", str(steps))
                    
                    # Log metrics
                    mlflow.log_metric("registered", 1)
                    
                    # Log the model to MLflow (without registering to avoid version issues)
                    artifact_path = "model"
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=artifact_path
                    )
                    
                    print(f"✓ Model logged to MLflow")
                    print(f"  Run ID: {run.info.run_id}")
                    
                    # Register model to Model Registry
                    model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
                    
                    try:
                        # Create or get model in registry
                        try:
                            client.create_registered_model(model_name)
                            print(f"✓ Created new registered model: {model_name}")
                        except Exception:
                            print(f"  Model '{model_name}' already exists in registry")
                        
                        # Create a new version
                        mv = client.create_model_version(
                            name=model_name,
                            source=model_uri,
                            run_id=run.info.run_id
                        )
                        print(f"✓ Created model version: {mv.version}")
                        
                        # Transition to Production stage
                        client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage="Production"
                        )
                        print(f"✓ Model transitioned to Production stage")
                        
                    except Exception as reg_error:
                        print(f"⚠ Could not register to Model Registry: {reg_error}")
                        print(f"  Model is still logged and can be accessed via run ID")
                    
            except Exception as e:
                print(f"✗ Error processing {model_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠ Not found: {model_path}")
    
    # Also try to register model_artifact.joblib using a simpler approach
    print("\n" + "-" * 60)
    print("📦 Attempting to register model_artifact.joblib...")
    
    if os.path.exists("model_artifact.joblib"):
        try:
            with mlflow.start_run(run_name="ChurnModel_Ensemble_Upload") as run:
                # Log the file as an artifact directly
                mlflow.log_artifact("model_artifact.joblib", artifact_path="models")
                mlflow.log_param("model_name", "ChurnModel_Ensemble")
                mlflow.log_param("artifact_type", "joblib_file")
                mlflow.log_param("upload_date", datetime.now().isoformat())
                mlflow.log_metric("uploaded", 1)
                
                print(f"✓ model_artifact.joblib uploaded as artifact")
                print(f"  Run ID: {run.info.run_id}")
                print(f"  View at: {mlflow_uri}/#/experiments/1/runs/{run.info.run_id}")
        except Exception as e:
            print(f"✗ Error uploading model_artifact.joblib: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model registration complete!")
    print("=" * 60)
    print("\n📌 Access your models:")
    print(f"   - MLflow UI: {mlflow_uri}")
    print("   - Experiments tab: View logged runs")
    print("   - Models tab: View registered models")


if __name__ == "__main__":
    register_model_to_mlflow()
