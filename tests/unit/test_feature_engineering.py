"""
Unit Tests for Feature Engineering Module
Tests for High-Cardinality patterns: Hashing, Embeddings, Feature Cross
"""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import (
    EmbeddingEncoder,
    FeatureCrosser,
    HashingEncoder,
)


class TestHashingEncoder:
    """Tests for HashingEncoder (Hashed Feature Pattern)"""

    def test_init(self):
        """Test HashingEncoder initialization"""
        encoder = HashingEncoder(n_features=16, col="category")
        assert encoder.n_features == 16
        assert encoder.col == "category"

    def test_fit_returns_self(self):
        """Test that fit returns self for pipeline compatibility"""
        encoder = HashingEncoder(n_features=8)
        df = pd.DataFrame({"category": ["a", "b", "c"]})
        result = encoder.fit(df)
        assert result is encoder

    def test_transform_output_shape(self):
        """Test output shape matches n_features"""
        encoder = HashingEncoder(n_features=16, col="category")
        df = pd.DataFrame({"category": ["apple", "banana", "cherry", "apple"]})
        encoder.fit(df)
        result = encoder.transform(df)

        assert result.shape == (4, 16)

    def test_transform_handles_unseen_categories(self):
        """Test that unseen categories are handled without error"""
        encoder = HashingEncoder(n_features=8, col="category")
        train_df = pd.DataFrame({"category": ["a", "b", "c"]})
        test_df = pd.DataFrame({"category": ["d", "e", "f"]})  # Unseen

        encoder.fit(train_df)
        result = encoder.transform(test_df)

        assert result.shape == (3, 8)
        assert not np.isnan(result).any()

    def test_same_input_same_hash(self):
        """Test that same input produces same hash"""
        encoder = HashingEncoder(n_features=32, col="category")
        df = pd.DataFrame({"category": ["test", "test", "other"]})

        encoder.fit(df)
        result = encoder.transform(df)

        # First two rows should be identical
        np.testing.assert_array_equal(result[0], result[1])
        # Third row should be different
        assert not np.array_equal(result[0], result[2])


class TestEmbeddingEncoder:
    """Tests for EmbeddingEncoder (Embeddings Pattern)"""

    def test_init(self):
        """Test EmbeddingEncoder initialization"""
        encoder = EmbeddingEncoder(embedding_dim=16, col="category")
        assert encoder.embedding_dim == 16
        assert encoder.col == "category"

    def test_fit_creates_embeddings(self):
        """Test that fit creates embedding vectors for each category"""
        encoder = EmbeddingEncoder(embedding_dim=8, col="category")
        df = pd.DataFrame({"category": ["a", "b", "c", "a"]})

        encoder.fit(df)

        assert len(encoder.embeddings_) == 3  # 3 unique categories
        assert all(vec.shape == (8,) for vec in encoder.embeddings_.values())

    def test_transform_output_shape(self):
        """Test transform output shape"""
        encoder = EmbeddingEncoder(embedding_dim=16, col="category")
        df = pd.DataFrame({"category": ["a", "b", "c"]})

        encoder.fit(df)
        result = encoder.transform(df)

        assert result.shape == (3, 16)

    def test_unknown_category_returns_zero_vector(self):
        """Test that unknown categories return zero vector"""
        encoder = EmbeddingEncoder(embedding_dim=8, col="category")
        train_df = pd.DataFrame({"category": ["a", "b"]})
        test_df = pd.DataFrame({"category": ["unknown"]})

        encoder.fit(train_df)
        result = encoder.transform(test_df)

        np.testing.assert_array_equal(result[0], np.zeros(8))

    def test_xavier_initialization_bounds(self):
        """Test that embeddings are within Xavier initialization bounds"""
        encoder = EmbeddingEncoder(embedding_dim=32, col="category")
        df = pd.DataFrame({"category": [f"cat_{i}" for i in range(100)]})

        encoder.fit(df)

        limit = np.sqrt(6 / (1 + 32))
        for vec in encoder.embeddings_.values():
            assert np.all(vec >= -limit) and np.all(vec <= limit)


class TestFeatureCrosser:
    """Tests for FeatureCrosser (Feature Cross Pattern)"""

    def test_init(self):
        """Test FeatureCrosser initialization"""
        crosser = FeatureCrosser(interactions=[("a", "b")])
        assert crosser.interactions == [("a", "b")]

    def test_creates_crossed_features(self):
        """Test that crossed features are created"""
        crosser = FeatureCrosser(interactions=[("gender", "contract")])
        df = pd.DataFrame(
            {
                "gender": ["Male", "Female", "Male"],
                "contract": ["Monthly", "Yearly", "Monthly"],
            }
        )

        result = crosser.fit_transform(df)

        assert "gender_x_contract" in result.columns
        assert result["gender_x_contract"].iloc[0] == "Male_Monthly"
        assert result["gender_x_contract"].iloc[1] == "Female_Yearly"

    def test_multiple_crosses(self):
        """Test multiple feature crosses"""
        crosser = FeatureCrosser(interactions=[("a", "b"), ("b", "c")])
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"], "c": ["p", "q"]})

        result = crosser.fit_transform(df)

        assert "a_x_b" in result.columns
        assert "b_x_c" in result.columns

    def test_missing_column_ignored(self):
        """Test that missing columns don't cause error"""
        crosser = FeatureCrosser(interactions=[("a", "missing")])
        df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]})

        result = crosser.fit_transform(df)

        assert "a_x_missing" not in result.columns

    def test_non_dataframe_passthrough(self):
        """Test that non-DataFrame input passes through"""
        crosser = FeatureCrosser(interactions=[("a", "b")])
        arr = np.array([[1, 2], [3, 4]])

        result = crosser.fit_transform(arr)

        np.testing.assert_array_equal(result, arr)
