"""
Ultimate Churn Prediction Model
Combines XGBoost, LightGBM, and Random Forest with optimized hyperparameters
and weighted ensemble for maximum predictive performance.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class UltimateChurnPredictor:
    """
    Ultimate Churn Prediction Model combining XGBoost, LightGBM, and Random Forest
    with optimized ensemble weights for maximum performance.
    """

    def __init__(self, optimize_hyperparams: bool = True):
        self.optimize_hyperparams = optimize_hyperparams
        self.models = {}
        self.weights = None
        self.is_fitted = False
        self.feature_importances_ = None
        self.training_metrics = {}

    def _get_optimized_xgboost(self) -> XGBClassifier:
        """
        Returns XGBoost with optimized hyperparameters for churn prediction.
        """
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    def _get_optimized_lightgbm(self) -> LGBMClassifier:
        """
        Returns LightGBM with optimized hyperparameters for churn prediction.
        """
        return LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

    def _get_optimized_random_forest(self) -> RandomForestClassifier:
        """
        Returns Random Forest with optimized hyperparameters for churn prediction.
        """
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    def _optimize_weights(
        self, predictions: np.ndarray, y_true: np.ndarray
    ) -> np.ndarray:
        """
        Optimize ensemble weights to maximize accuracy using grid search.
        """
        n_models = predictions.shape[1]
        best_weights = np.ones(n_models) / n_models
        best_acc = 0

        # Fine-grained grid search
        from itertools import product

        weight_options = [0.2, 0.25, 0.3, 0.33, 0.35, 0.4, 0.45, 0.5]

        for w1, w2 in product(weight_options, repeat=2):
            w3 = 1.0 - w1 - w2
            if w3 < 0.1 or w3 > 0.6:
                continue

            weights = np.array([w1, w2, w3])
            weighted_proba = np.average(predictions, axis=1, weights=weights)
            weighted_pred = (weighted_proba > 0.5).astype(int)

            acc = accuracy_score(y_true, weighted_pred)

            if acc > best_acc:
                best_acc = acc
                best_weights = weights

        return best_weights

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the ultimate ensemble model.

        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        X_val : array-like, optional
            Validation features for weight optimization
        y_val : array-like, optional
            Validation labels for weight optimization
        """
        print("\n" + "=" * 60)
        print("Training Ultimate Churn Predictor")
        print("=" * 60)

        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize models
        self.models = {
            "xgboost": self._get_optimized_xgboost(),
            "lightgbm": self._get_optimized_lightgbm(),
            "random_forest": self._get_optimized_random_forest(),
        }

        # Train each model
        for name, model in self.models.items():
            print(f"\n  Training {name}...")
            model.fit(X, y)
            print(f"    ✓ {name} trained successfully")

        # Optimize weights using validation set or cross-validation
        if X_val is not None and y_val is not None:
            print("\n  Optimizing ensemble weights using validation set...")
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)

            val_predictions = self._get_base_predictions(X_val)
            self.weights = self._optimize_weights(val_predictions, y_val)
        else:
            print("\n  Optimizing ensemble weights using cross-validation...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            oof_predictions = np.zeros((len(y), 3))

            for idx, (name, model) in enumerate(self.models.items()):
                # Get out-of-fold predictions
                temp_model = self._get_model_by_name(name)
                oof_pred = cross_val_predict(
                    temp_model, X, y, cv=skf, method="predict_proba"
                )[:, 1]
                oof_predictions[:, idx] = oof_pred

            self.weights = self._optimize_weights(oof_predictions, y)

        print(f"\n  Optimized Weights:")
        for name, weight in zip(self.models.keys(), self.weights):
            print(f"    {name}: {weight:.4f}")

        # Compute feature importances (weighted average)
        self._compute_feature_importances(X.shape[1])

        # Mark as fitted before computing training metrics
        self.is_fitted = True

        # Compute training metrics
        train_pred = self.predict(X)
        train_proba = self.predict_proba(X)[:, 1]

        self.training_metrics = {
            "accuracy": accuracy_score(y, train_pred),
            "precision": precision_score(y, train_pred),
            "recall": recall_score(y, train_pred),
            "f1": f1_score(y, train_pred),
            "roc_auc": roc_auc_score(y, train_proba),
        }

        print(f"\n  Training Metrics:")
        print(f"    Accuracy:  {self.training_metrics['accuracy']:.4f}")
        print(f"    Precision: {self.training_metrics['precision']:.4f}")
        print(f"    Recall:    {self.training_metrics['recall']:.4f}")
        print(f"    F1 Score:  {self.training_metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {self.training_metrics['roc_auc']:.4f}")

        self.is_fitted = True
        print("\n  ✓ Ultimate Churn Predictor trained successfully!")

        return self

    def _get_model_by_name(self, name: str):
        """Get a fresh model instance by name."""
        if name == "xgboost":
            return self._get_optimized_xgboost()
        elif name == "lightgbm":
            return self._get_optimized_lightgbm()
        elif name == "random_forest":
            return self._get_optimized_random_forest()

    def _get_base_predictions(self, X) -> np.ndarray:
        """Get probability predictions from all base models."""
        X = np.asarray(X)
        predictions = []

        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        return np.column_stack(predictions)

    def _compute_feature_importances(self, n_features: int):
        """Compute weighted feature importances."""
        importances = np.zeros(n_features)

        for (name, model), weight in zip(self.models.items(), self.weights):
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                if len(imp) == n_features:
                    importances += weight * imp

        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()

        self.feature_importances_ = importances

    def predict(self, X) -> np.ndarray:
        """
        Predict churn labels.

        Parameters:
        -----------
        X : array-like
            Features

        Returns:
        --------
        predictions : array
            Binary predictions (0 = No Churn, 1 = Churn)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict churn probabilities.

        Parameters:
        -----------
        X : array-like
            Features

        Returns:
        --------
        probabilities : array of shape (n_samples, 2)
            Probability of [No Churn, Churn]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        base_preds = self._get_base_predictions(X)
        weighted_proba = np.average(base_preds, axis=1, weights=self.weights)

        return np.column_stack([1 - weighted_proba, weighted_proba])

    def evaluate(self, X, y, verbose: bool = True) -> dict:
        """
        Evaluate model performance on test data.

        Parameters:
        -----------
        X : array-like
            Test features
        y : array-like
            Test labels
        verbose : bool
            Print results

        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        X = np.asarray(X)
        y = np.asarray(y)

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        if verbose:
            print("\n" + "=" * 60)
            print("Ultimate Churn Predictor - Evaluation Results")
            print("=" * 60)
            print(f"\n  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1 Score:    {metrics['f1']:.4f}")
            print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {tn:5d}  |  FP: {fp:5d}")
            print(f"    FN: {fn:5d}  |  TP: {tp:5d}")
            print("\n" + "-" * 60)
            print("  Classification Report:")
            print("-" * 60)
            print(classification_report(y, y_pred, target_names=["No Churn", "Churn"]))

        return metrics

    def get_individual_model_scores(self, X, y) -> pd.DataFrame:
        """
        Get performance scores for each individual model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            results.append(
                {
                    "model": name,
                    "weight": self.weights[list(self.models.keys()).index(name)],
                    "accuracy": accuracy_score(y, y_pred),
                    "precision": precision_score(y, y_pred),
                    "recall": recall_score(y, y_pred),
                    "f1": f1_score(y, y_pred),
                    "roc_auc": roc_auc_score(y, y_proba),
                }
            )

        # Add ensemble
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]

        results.append(
            {
                "model": "ENSEMBLE",
                "weight": 1.0,
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "f1": f1_score(y, y_pred),
                "roc_auc": roc_auc_score(y, y_proba),
            }
        )

        return pd.DataFrame(results)

    def get_feature_importance_df(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importances as a DataFrame.
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet.")

        if feature_names is None:
            feature_names = [
                f"feature_{i}" for i in range(len(self.feature_importances_))
            ]

        df = pd.DataFrame(
            {
                "feature": feature_names[: len(self.feature_importances_)],
                "importance": self.feature_importances_,
            }
        )

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, filepath: str):
        """Save the model to disk."""
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        state = {
            "models": self.models,
            "weights": self.weights,
            "is_fitted": self.is_fitted,
            "feature_importances_": self.feature_importances_,
            "training_metrics": self.training_metrics,
        }

        joblib.dump(state, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load the model from disk."""
        state = joblib.load(filepath)

        self.models = state["models"]
        self.weights = state["weights"]
        self.is_fitted = state["is_fitted"]
        self.feature_importances_ = state["feature_importances_"]
        self.training_metrics = state["training_metrics"]

        print(f"Model loaded from {filepath}")
        return self

    def score(self, X, y) -> float:
        """Return accuracy score (sklearn compatible)."""
        return accuracy_score(y, self.predict(X))

    def __sklearn_is_fitted__(self):
        """Check if fitted (sklearn compatible)."""
        return self.is_fitted


def train_ultimate_model(
    data_path: str, save_path: str = "ultimate_churn_model.joblib"
):
    """
    Train the ultimate churn prediction model from data.

    Parameters:
    -----------
    data_path : str
        Path to the Telco Churn dataset
    save_path : str
        Path to save the trained model
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    print("\n" + "#" * 70)
    print("#" + " " * 15 + "ULTIMATE CHURN PREDICTION MODEL" + " " * 15 + "#")
    print("#" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv(data_path)

    # Preprocessing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.dropna(subset=["Churn"])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"    Loaded {len(df)} samples")
    print(f"    Churn rate: {y.mean()*100:.1f}%")

    # Build preprocessor
    print("\n[2/4] Building feature preprocessor...")

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

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
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
    )

    print(f"    Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Transform features
    print("\n[3/4] Transforming features...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_val_t = X_val_t.toarray()
        X_test_t = X_test_t.toarray()

    print(f"    Feature dimension: {X_train_t.shape[1]}")

    # Train model
    print("\n[4/4] Training Ultimate Churn Predictor...")
    model = UltimateChurnPredictor()
    model.fit(X_train_t, y_train, X_val_t, y_val)

    # Evaluate
    print("\n" + "=" * 70)
    print("FINAL TEST SET EVALUATION")
    print("=" * 70)

    test_metrics = model.evaluate(X_test_t, y_test)

    # Compare individual models vs ensemble
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL vs ENSEMBLE COMPARISON")
    print("=" * 70)

    comparison_df = model.get_individual_model_scores(X_test_t, y_test)
    print("\n" + comparison_df.to_string(index=False))

    # Save model and preprocessor
    model.save(save_path)
    joblib.dump(preprocessor, save_path.replace(".joblib", "_preprocessor.joblib"))

    print("\n" + "#" * 70)
    print("#" + " " * 20 + "TRAINING COMPLETE" + " " * 20 + "#")
    print("#" * 70)
    print(f"\n  Model saved to: {save_path}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")

    return model, preprocessor, test_metrics


if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"

    model, preprocessor, metrics = train_ultimate_model(
        data_url, save_path="ultimate_churn_model.joblib"
    )
