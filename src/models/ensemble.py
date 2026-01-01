import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier

from src.models.base_model import BaseModel


class EnsembleModel(BaseModel):
    """
    Ensemble model supporting Voting, Stacking, and Bagging.
    Integrates XGBoost, LightGBM, and RandomForest.
    """

    def __init__(self, ensemble_type="voting", rebalancing_strategy=None):
        super().__init__(rebalancing_strategy)
        self.ensemble_type = ensemble_type
        # Define base models
        self.base_models = [
            ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
            ("lgbm", LGBMClassifier(random_state=42, verbose=-1)),
            ("rf", RandomForestClassifier(random_state=42)),
        ]
        self.model = None
        self._is_fitted = False

    def fit(self, X, y):
        # Convert to plain numpy array to strip any feature names metadata
        # This ensures models like LGBMClassifier don't store feature names
        if hasattr(X, "values"):
            X = X.values
        X = np.array(X, dtype=np.float64)
        y = np.asarray(y)

        # Apply rebalancing
        X_res, y_res = self._apply_rebalancing(X, y)

        # Ensure plain numpy array format after rebalancing
        if hasattr(X_res, "values"):
            X_res = X_res.values
        X_res = np.array(X_res, dtype=np.float64)

        print(f"Training Ensemble: {self.ensemble_type}")

        if self.ensemble_type == "voting":
            # Soft voting averages probabilities
            self.model = VotingClassifier(estimators=self.base_models, voting="soft")
        elif self.ensemble_type == "stacking":
            # Meta-learner: XGBoost
            self.model = StackingClassifier(
                estimators=self.base_models,
                final_estimator=XGBClassifier(eval_metric="logloss", random_state=42),
                cv=3,
            )
        elif self.ensemble_type == "bagging":
            # Bagging with XGBoost
            self.model = BaggingClassifier(
                estimator=XGBClassifier(eval_metric="logloss", random_state=42),
                n_estimators=10,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")

        self.model.fit(X_res, y_res)
        self._is_fitted = True
        return self

    def predict(self, X):
        # Convert to numpy array for consistent format
        X = np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            return self.model.predict(X)

    def predict_proba(self, X):
        # Convert to numpy array for consistent format
        X = np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            return self.model.predict_proba(X)

    def score(self, X, y):
        """Return the accuracy score on the given test data and labels."""
        # Convert to numpy array for consistent format
        X = np.asarray(X)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="X does not have valid feature names"
            )
            return self.model.score(X, y)

    def __sklearn_is_fitted__(self):
        """Check if the model is fitted."""
        return self._is_fitted
