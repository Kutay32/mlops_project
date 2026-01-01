import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher


class HashingEncoder(BaseEstimator, TransformerMixin):
    """
    Converts high-cardinality categorical features into fixed buckets
    using feature hashing. Handles unseen categories at inference time.
    """

    def __init__(self, n_features: int = 256, col: str = None):
        self.n_features = n_features
        self.col = col
        # Use 'dict' to hash the full category string as a token
        self.hasher = FeatureHasher(n_features=n_features, input_type="dict")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Determine input data
        if isinstance(X, pd.DataFrame):
            if self.col and self.col in X.columns:
                data = X[self.col]
            else:
                data = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            data = X.flatten()
        else:
            data = X  # Assume list/series

        # Convert to list of dicts for FeatureHasher
        key = self.col if self.col else "feat"
        dicts = [{key: str(x)} for x in data]

        return self.hasher.transform(dicts).toarray()


class EmbeddingEncoder(BaseEstimator, TransformerMixin):
    """
    Maps high-cardinality categorical data into dense embedding space.
    Uses Xavier initialization for embedding vectors.
    """

    def __init__(self, embedding_dim: int = 32, col: str = None):
        self.embedding_dim = embedding_dim
        self.col = col
        self.embeddings_ = {}
        self.unknown_vector_ = None

    def fit(self, X, y=None):
        # Extract unique values
        if isinstance(X, pd.DataFrame):
            if self.col and self.col in X.columns:
                vals = X[self.col]
            else:
                vals = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            vals = X.flatten()
        else:
            vals = X

        categories = pd.unique(vals)
        categories = [str(x) for x in categories]
        n_cats = len(categories)

        # Xavier Initialization
        limit = np.sqrt(6 / (1 + self.embedding_dim))
        vectors = np.random.uniform(-limit, limit, (n_cats, self.embedding_dim))

        self.embeddings_ = {cat: vec for cat, vec in zip(categories, vectors)}
        self.unknown_vector_ = np.zeros(self.embedding_dim)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if self.col and self.col in X.columns:
                vals = X[self.col]
            else:
                vals = X.iloc[:, 0]
        elif isinstance(X, np.ndarray):
            vals = X.flatten()
        else:
            vals = X

        def get_vec(val):
            return self.embeddings_.get(str(val), self.unknown_vector_)

        vecs = [get_vec(x) for x in vals]
        return np.array(vecs)


class FeatureCrosser(BaseEstimator, TransformerMixin):
    """
    Capture feature interactions.
    Note: Requires DataFrame input with column names.
    Usage in Pipeline: Apply before steps that strip column names (like standard Scalers).
    """

    def __init__(self, interactions: list = None):
        self.interactions = interactions if interactions else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            # If input is not DF, we can't easily cross by name.
            # We return as is or raise warning.
            return X

        X = X.copy()
        for col1, col2 in self.interactions:
            if col1 in X.columns and col2 in X.columns:
                new_col = f"{col1}_x_{col2}"
                X[new_col] = X[col1].astype(str) + "_" + X[col2].astype(str)
        return X
