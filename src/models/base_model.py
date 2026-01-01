from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import RandomUnderSampler


class BaseModel(ABC):
    """
    Abstract base model class ensuring consistent interface and
    common functionality like data rebalancing.
    """

    def __init__(self, rebalancing_strategy: str = None):
        self.rebalancing_strategy = rebalancing_strategy
        self.model = None

    def _apply_rebalancing(self, X, y):
        """
        Apply rebalancing strategy to the training data.
        Strategies:
        - SMOTE: Synthetic Minority Over-sampling
        - ADASYN: Adaptive Synthetic Sampling
        - undersample: Random majority reduction
        - None: No rebalancing

        Note: Class weights are typically handled by the model hyperparameters,
        not by resampling.
        """
        # Convert to numpy arrays to ensure consistent format
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        print(f"Applying rebalancing strategy: {self.rebalancing_strategy}")
        if self.rebalancing_strategy == "SMOTE":
            sampler = SMOTE(random_state=42)
            return sampler.fit_resample(X, y)
        elif self.rebalancing_strategy == "ADASYN":
            try:
                sampler = ADASYN(random_state=42)
                return sampler.fit_resample(X, y)
            except ValueError:
                # Fallback if ADASYN fails (e.g. not enough neighbors)
                print("ADASYN failed (likely sparse minority), falling back to SMOTE")
                sampler = SMOTE(random_state=42)
                return sampler.fit_resample(X, y)
        elif self.rebalancing_strategy == "undersample":
            sampler = RandomUnderSampler(random_state=42)
            return sampler.fit_resample(X, y)

        return X, y

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass
