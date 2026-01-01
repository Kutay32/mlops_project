"""
Prefect Workflow Orchestrator
DAG-based training pipeline with task dependencies
"""
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd
import numpy as np
import mlflow
import joblib
from typing import Dict, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


# ═══════════════════════════════════════════════════════════════
# Data Loading Tasks
# ═══════════════════════════════════════════════════════════════

@task(
    name="Load Data",
    description="Load dataset from source",
    retries=3,
    retry_delay_seconds=10,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1)
)
def load_data(data_path: str = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        data_path: Path to CSV file (optional, downloads from GitHub if not provided)
        
    Returns:
        Raw DataFrame
    """
    logger = get_run_logger()
    
    if data_path:
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        logger.info(f"Downloading data from GitHub: {url}")
        df = pd.read_csv(url)
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


@task(
    name="Validate Data",
    description="Validate raw data quality"
)
def validate_data(df: pd.DataFrame) -> Dict:
    """
    Validate data quality before processing.
    
    Returns:
        Validation report dictionary
    """
    logger = get_run_logger()
    
    validation_report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "target_distribution": df['Churn'].value_counts().to_dict() if 'Churn' in df.columns else {}
    }
    
    # Check for critical issues
    issues = []
    
    if len(df) == 0:
        issues.append("Empty dataset")
    
    if df.duplicated().sum() > len(df) * 0.1:
        issues.append("High duplicate rate (>10%)")
    
    missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_pct > 0.2:
        issues.append(f"High missing value rate ({missing_pct:.1%})")
    
    validation_report["issues"] = issues
    validation_report["passed"] = len(issues) == 0
    
    if issues:
        logger.warning(f"Validation issues: {issues}")
    else:
        logger.info("Data validation passed")
    
    return validation_report


# ═══════════════════════════════════════════════════════════════
# Feature Engineering Tasks
# ═══════════════════════════════════════════════════════════════

@task(
    name="Preprocess Data",
    description="Clean and preprocess raw data"
)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw data.
    
    Steps:
    - Handle missing values
    - Convert data types
    - Remove duplicates
    """
    logger = get_run_logger()
    
    df = df.copy()
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert target
    if 'Churn' in df.columns:
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


@task(
    name="Engineer Features",
    description="Create derived features"
)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features.
    
    New features:
    - AvgMonthlyCharges
    - tenure_group
    - service_count
    """
    logger = get_run_logger()
    
    df = df.copy()
    
    # Average monthly charges
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Tenure groups
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
    )
    
    # Service count
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    service_count = 0
    for col in service_cols:
        if col in df.columns:
            service_count += (df[col] != 'No').astype(int)
    df['service_count'] = service_count
    
    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


@task(
    name="Encode Features",
    description="Encode categorical features"
)
def encode_features(
    df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Encode categorical features and prepare for training.
    
    Returns:
        X: Feature matrix
        y: Target array
        encoders: Dictionary of fitted encoders
    """
    logger = get_run_logger()
    
    df = df.copy()
    encoders = {}
    
    # Separate target
    y = df['Churn'].values if 'Churn' in df.columns else None
    df = df.drop('Churn', axis=1, errors='ignore')
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categoricals
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    X = df.values.astype(np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    encoders['scaler'] = scaler
    encoders['feature_names'] = df.columns.tolist()
    
    logger.info(f"Encoding complete. X shape: {X.shape}")
    return X, y, encoders


# ═══════════════════════════════════════════════════════════════
# Model Training Tasks
# ═══════════════════════════════════════════════════════════════

@task(
    name="Split Data",
    description="Split data into train/test sets"
)
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    logger = get_run_logger()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@task(
    name="Train RandomForest",
    description="Train Random Forest classifier"
)
def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict = None
) -> RandomForestClassifier:
    """Train Random Forest model."""
    logger = get_run_logger()
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    params = {**default_params, **(params or {})}
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("Random Forest training complete")
    return model


@task(
    name="Train XGBoost",
    description="Train XGBoost classifier"
)
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict = None
) -> XGBClassifier:
    """Train XGBoost model."""
    logger = get_run_logger()
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    params = {**default_params, **(params or {})}
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("XGBoost training complete")
    return model


@task(
    name="Train LightGBM",
    description="Train LightGBM classifier"
)
def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict = None
) -> LGBMClassifier:
    """Train LightGBM model."""
    logger = get_run_logger()
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    params = {**default_params, **(params or {})}
    
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info("LightGBM training complete")
    return model


# ═══════════════════════════════════════════════════════════════
# Evaluation Tasks
# ═══════════════════════════════════════════════════════════════

@task(
    name="Evaluate Model",
    description="Evaluate model performance"
)
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict:
    """
    Evaluate model and compute metrics.
    
    Returns:
        Dictionary with all evaluation metrics
    """
    logger = get_run_logger()
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    return metrics


@task(
    name="Select Best Model",
    description="Select best model based on metrics"
)
def select_best_model(
    models: Dict[str, Any],
    metrics: Dict[str, Dict],
    selection_metric: str = 'f1_score'
) -> Tuple[str, Any]:
    """
    Select the best model based on a metric.
    
    Returns:
        (best_model_name, best_model)
    """
    logger = get_run_logger()
    
    best_name = max(metrics, key=lambda x: metrics[x][selection_metric])
    best_model = models[best_name]
    best_score = metrics[best_name][selection_metric]
    
    logger.info(f"Best model: {best_name} with {selection_metric}={best_score:.4f}")
    return best_name, best_model


# ═══════════════════════════════════════════════════════════════
# MLflow Tasks
# ═══════════════════════════════════════════════════════════════

@task(
    name="Log to MLflow",
    description="Log model and metrics to MLflow"
)
def log_to_mlflow(
    model: Any,
    metrics: Dict,
    model_name: str,
    experiment_name: str = "churn_prefect_pipeline"
):
    """Log model and metrics to MLflow."""
    logger = get_run_logger()
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"prefect_{model_name}"):
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"Prefect_{model_name}"
        )
        
        logger.info(f"Logged {model_name} to MLflow")


@task(
    name="Save Artifacts",
    description="Save model and encoders to disk"
)
def save_artifacts(
    model: Any,
    encoders: Dict,
    model_name: str,
    output_dir: str = "model_artifacts"
):
    """Save model and preprocessing artifacts."""
    logger = get_run_logger()
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    encoders_path = os.path.join(output_dir, "encoders.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(encoders, encoders_path)
    
    logger.info(f"Saved artifacts to {output_dir}")


# ═══════════════════════════════════════════════════════════════
# Main Flow
# ═══════════════════════════════════════════════════════════════

@flow(
    name="Churn Model Training Pipeline",
    description="End-to-end ML pipeline for customer churn prediction",
    version="1.0.0"
)
def training_pipeline(
    data_path: Optional[str] = None,
    experiment_name: str = "churn_prefect_pipeline",
    log_to_mlflow_enabled: bool = True
):
    """
    Main training pipeline flow.
    
    DAG Structure:
    load_data -> validate_data -> preprocess_data -> engineer_features
        -> encode_features -> split_data
        -> train_rf/train_xgb/train_lgbm (parallel)
        -> evaluate (parallel)
        -> select_best -> log_to_mlflow -> save_artifacts
    """
    logger = get_run_logger()
    logger.info("Starting Churn Model Training Pipeline")
    
    # ─── Data Loading & Validation ───
    raw_data = load_data(data_path)
    validation = validate_data(raw_data)
    
    if not validation["passed"]:
        logger.warning("Data validation failed, but continuing...")
    
    # ─── Feature Engineering ───
    preprocessed = preprocess_data(raw_data)
    engineered = engineer_features(preprocessed)
    X, y, encoders = encode_features(engineered)
    
    # ─── Train/Test Split ───
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # ─── Model Training (Parallel) ───
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)
    
    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgbm_model
    }
    
    # ─── Evaluation (Parallel) ───
    metrics = {}
    for name, model in models.items():
        metrics[name] = evaluate_model(model, X_test, y_test, name)
    
    # ─── Select Best Model ───
    best_name, best_model = select_best_model(models, metrics)
    
    # ─── Log to MLflow ───
    if log_to_mlflow_enabled:
        for name, model in models.items():
            log_to_mlflow(model, metrics[name], name, experiment_name)
    
    # ─── Save Artifacts ───
    save_artifacts(best_model, encoders, best_name)
    
    logger.info(f"Pipeline complete! Best model: {best_name}")
    
    return {
        "best_model": best_name,
        "metrics": metrics,
        "validation": validation
    }


# ═══════════════════════════════════════════════════════════════
# Data Quality Flow (Great Expectations Alternative)
# ═══════════════════════════════════════════════════════════════

@flow(name="Data Quality Check")
def data_quality_flow(data_path: str = None):
    """
    Separate flow for data quality checks.
    Can be scheduled independently.
    """
    data = load_data(data_path)
    validation = validate_data(data)
    return validation


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run the pipeline
    result = training_pipeline()
    print(f"\nPipeline Result: {result}")
