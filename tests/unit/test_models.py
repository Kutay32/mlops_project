"""
Unit Tests for Model Components
Tests for Ensemble, Rebalancing, and Reframing patterns
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TestEnsembleModel:
    """Tests for EnsembleModel (Ensembles Pattern)"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data"""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_redundant=5, n_classes=2, random_state=42
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_voting_ensemble_fit(self, sample_data):
        """Test VotingClassifier ensemble fits correctly"""
        from src.models.ensemble import EnsembleModel
        
        X_train, X_test, y_train, y_test = sample_data
        model = EnsembleModel(ensemble_type='voting')
        
        model.fit(X_train, y_train)
        
        assert model._is_fitted
        assert model.model is not None
    
    def test_voting_ensemble_predict(self, sample_data):
        """Test VotingClassifier predictions"""
        from src.models.ensemble import EnsembleModel
        
        X_train, X_test, y_train, y_test = sample_data
        model = EnsembleModel(ensemble_type='voting')
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})
    
    def test_stacking_ensemble(self, sample_data):
        """Test StackingClassifier ensemble"""
        from src.models.ensemble import EnsembleModel
        
        X_train, X_test, y_train, y_test = sample_data
        model = EnsembleModel(ensemble_type='stacking')
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        
        assert 0 <= score <= 1
    
    def test_bagging_ensemble(self, sample_data):
        """Test BaggingClassifier ensemble"""
        from src.models.ensemble import EnsembleModel
        
        X_train, X_test, y_train, y_test = sample_data
        model = EnsembleModel(ensemble_type='bagging')
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)
        
        assert proba.shape == (len(y_test), 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    
    def test_invalid_ensemble_type_raises(self, sample_data):
        """Test that invalid ensemble type raises error"""
        from src.models.ensemble import EnsembleModel
        
        X_train, _, y_train, _ = sample_data
        model = EnsembleModel(ensemble_type='invalid')
        
        with pytest.raises(ValueError):
            model.fit(X_train, y_train)


class TestRebalancing:
    """Tests for Rebalancing Pattern (SMOTE, ADASYN, Undersample)"""
    
    @pytest.fixture
    def imbalanced_data(self):
        """Create imbalanced dataset"""
        X, y = make_classification(
            n_samples=1000, n_features=10, n_informative=5,
            n_classes=2, weights=[0.9, 0.1], random_state=42
        )
        return X, y
    
    def test_smote_rebalancing(self, imbalanced_data):
        """Test SMOTE rebalancing"""
        from src.models.ensemble import EnsembleModel
        
        X, y = imbalanced_data
        model = EnsembleModel(ensemble_type='voting', rebalancing_strategy='SMOTE')
        
        # Check class distribution before
        unique, counts = np.unique(y, return_counts=True)
        assert counts[0] > counts[1] * 5  # Highly imbalanced
        
        model.fit(X, y)
        assert model._is_fitted
    
    def test_undersample_rebalancing(self, imbalanced_data):
        """Test undersampling rebalancing"""
        from src.models.ensemble import EnsembleModel
        
        X, y = imbalanced_data
        model = EnsembleModel(ensemble_type='voting', rebalancing_strategy='undersample')
        
        model.fit(X, y)
        assert model._is_fitted
    
    def test_no_rebalancing(self, imbalanced_data):
        """Test without rebalancing"""
        from src.models.ensemble import EnsembleModel
        
        X, y = imbalanced_data
        model = EnsembleModel(ensemble_type='voting', rebalancing_strategy=None)
        
        model.fit(X, y)
        assert model._is_fitted


class TestProblemReframing:
    """Tests for Problem Reframing Pattern"""
    
    def test_regression_to_classification(self):
        """Test converting regression target to classification"""
        from src.models.reframing import RegressionToClassification
        
        reframer = RegressionToClassification(n_bins=5, strategy='quantile')
        y_continuous = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        reframer.fit(y_continuous)
        y_discrete = reframer.transform(y_continuous)
        
        assert len(np.unique(y_discrete)) <= 5
        assert y_discrete.shape == y_continuous.shape
    
    def test_bin_edges_accessible(self):
        """Test that bin edges are accessible after fit"""
        from src.models.reframing import RegressionToClassification
        
        reframer = RegressionToClassification(n_bins=3)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        reframer.fit(y)
        edges = reframer.get_bin_edges()
        
        assert edges is not None
        assert len(edges[0]) == 4  # n_bins + 1 edges
