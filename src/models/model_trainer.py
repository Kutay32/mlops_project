"""
Individual Model Training Module
Trains and evaluates multiple base models independently with comprehensive metrics.
"""

import json
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class ModelTrainer:
    """
    Trains multiple base models independently and tracks performance metrics.
    """

    def __init__(self, output_dir: str = "model_artifacts"):
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        self.trained_models = {}
        os.makedirs(output_dir, exist_ok=True)

    def get_base_models(self) -> dict:
        """
        Returns dictionary of base models with default hyperparameters.
        """
        return {
            "xgboost": XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            ),
            "lightgbm": LGBMClassifier(
                random_state=42,
                verbose=-1,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
            ),
            "random_forest": RandomForestClassifier(
                random_state=42, n_estimators=100, max_depth=10, min_samples_split=5
            ),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1
            ),
            "logistic_regression": LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            ),
            "adaboost": AdaBoostClassifier(
                random_state=42, n_estimators=100, learning_rate=0.1
            ),
        }

    def compute_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """
        Compute comprehensive classification metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def train_model(self, name: str, model, X_train, y_train, X_val, y_val) -> dict:
        """
        Train a single model and compute validation metrics.
        """
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        # Convert to numpy arrays
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        # Train
        start_time = datetime.now()
        model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()

        # Predict
        y_pred = model.predict(X_val)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]

        # Compute metrics
        metrics = self.compute_metrics(y_val, y_pred, y_proba)
        metrics["train_time_seconds"] = train_time
        metrics["model_name"] = name

        # Store trained model
        self.trained_models[name] = model

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if "roc_auc" in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Train Time: {train_time:.2f}s")

        return metrics

    def train_all_models(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        """
        Train all base models and return comparison dataframe.
        """
        models = self.get_base_models()
        results = []

        for name, model in models.items():
            try:
                metrics = self.train_model(name, model, X_train, y_train, X_val, y_val)
                results.append(metrics)
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue

        self.results = results

        # Create comparison dataframe
        df = pd.DataFrame(results)
        df = df.sort_values("f1", ascending=False)

        return df

    def cross_validate_models(self, X, y, cv=5) -> pd.DataFrame:
        """
        Perform cross-validation on all models.
        """
        print(f"\n{'='*50}")
        print(f"Cross-Validation (cv={cv})")
        print(f"{'='*50}")

        X = np.asarray(X)
        y = np.asarray(y)

        models = self.get_base_models()
        cv_results = []

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
                cv_results.append(
                    {
                        "model_name": name,
                        "cv_mean_f1": scores.mean(),
                        "cv_std_f1": scores.std(),
                        "cv_scores": scores.tolist(),
                    }
                )
                print(f"  {name}: F1 = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            except Exception as e:
                print(f"  Error CV {name}: {e}")
                continue

        return pd.DataFrame(cv_results).sort_values("cv_mean_f1", ascending=False)

    def save_models(self, prefix: str = "base_model"):
        """
        Save all trained models and their configurations.
        """
        saved_paths = {}

        for name, model in self.trained_models.items():
            path = os.path.join(self.output_dir, f"{prefix}_{name}.joblib")
            joblib.dump(model, path)
            saved_paths[name] = path
            print(f"Saved {name} to {path}")

        # Save results summary
        results_path = os.path.join(self.output_dir, f"{prefix}_results.json")
        with open(results_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_results = []
            for r in self.results:
                sr = {}
                for k, v in r.items():
                    if isinstance(v, np.floating):
                        sr[k] = float(v)
                    elif isinstance(v, np.integer):
                        sr[k] = int(v)
                    elif isinstance(v, np.ndarray):
                        sr[k] = v.tolist()
                    else:
                        sr[k] = v
                serializable_results.append(sr)
            json.dump(serializable_results, f, indent=2)

        print(f"Saved results to {results_path}")
        return saved_paths

    def load_model(self, name: str, prefix: str = "base_model"):
        """
        Load a previously saved model.
        """
        path = os.path.join(self.output_dir, f"{prefix}_{name}.joblib")
        if os.path.exists(path):
            return joblib.load(path)
        raise FileNotFoundError(f"Model not found: {path}")

    def get_best_model(self, metric: str = "f1") -> tuple:
        """
        Return the best performing model based on specified metric.
        """
        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models first.")

        best = max(self.results, key=lambda x: x.get(metric, 0))
        name = best["model_name"]
        return name, self.trained_models.get(name), best
