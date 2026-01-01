"""
Train script that uses MLflow for tracking and registry operations.

Usage example:
  MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/train_with_mlflow.py \
      --data-url "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv" \
      --registered-model-name "UltimateChurnModel"
"""

import os
import argparse
import warnings

import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from src.data.feature_engineering import HashingEncoder, EmbeddingEncoder
from src.models.model_trainer import ModelTrainer

warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_path: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    df = pd.read_csv(data_path)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop_duplicates()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = df.dropna(subset=['Churn'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                            'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                            'Contract', 'PaymentMethod', 'InternetService']

    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
        ],
        remainder='drop'
    )

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42)

    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert sparse to dense if necessary
    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()
        X_val_transformed = X_val_transformed.toarray()
        X_test_transformed = X_test_transformed.toarray()

    return X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-url', type=str, required=True)
    parser.add_argument('--registered-model-name', type=str, default=None)
    args = parser.parse_args()

    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            print(f"Using MLflow tracking at: {mlflow_uri}")
        except Exception as e:
            print(f"Could not set MLflow tracking URI: {e}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(args.data_url)

    trainer = ModelTrainer(output_dir="model_artifacts", use_mlflow=True, registered_model_name=args.registered_model_name)

    results_df = trainer.train_all_models(X_train, y_train, X_val, y_val)
    print(results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))

    trainer.save_models()

    # Evaluate best on test set
    best_name, best_model, best_metrics = trainer.get_best_model('f1')
    if best_model is not None:
        try:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            y_pred = best_model.predict(X_test)
            y_proba = None
            if hasattr(best_model, 'predict_proba'):
                y_proba = best_model.predict_proba(X_test)[:, 1]

            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
            }
            if y_proba is not None:
                test_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)

            print(f"Best model on test: {best_name} -> {test_metrics}")

            # Log final test metrics in a separate MLflow run
            try:
                import mlflow
                with mlflow.start_run(run_name=f"test_eval_{best_name}"):
                    for k, v in test_metrics.items():
                        mlflow.log_metric(k, float(v))
                    mlflow.set_tag('model_name', best_name)
            except Exception as e:
                print(f"Could not log test metrics to MLflow: {e}")

        except Exception as e:
            print(f"Failed to evaluate best model: {e}")

    print("Training complete.")


if __name__ == '__main__':
    main()
