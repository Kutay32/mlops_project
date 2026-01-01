import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer


class ProblemReframer(BaseEstimator, TransformerMixin):
    """Base class for problem reframing patterns."""

    pass


class RegressionToClassification(ProblemReframer):
    """
    Converts continuous target into discrete classes via bucketing.
    Outputs probability distributions over value ranges.
    """

    def __init__(
        self, n_bins: int = 5, strategy: str = "quantile", encode: str = "ordinal"
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins, strategy=strategy, encode=encode
        )

    def fit(self, y, sample_weight=None):
        # KBinsDiscretizer expects 2D array
        y = np.array(y).reshape(-1, 1)
        self.discretizer.fit(y, sample_weight=sample_weight)
        return self

    def transform(self, y):
        y = np.array(y).reshape(-1, 1)
        # Flatten if ordinal, otherwise keep 2D (onehot)
        res = self.discretizer.transform(y)
        if self.encode == "ordinal":
            return res.flatten()
        return res

    def get_bin_edges(self):
        return self.discretizer.bin_edges_
