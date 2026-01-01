"""
Advanced Ensemble Module
Implements weighted averaging, stacking, and blending techniques for ensemble optimization.
"""
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import joblib
import os

warnings.filterwarnings("ignore")


class AdvancedEnsemble:
    """
    Advanced ensemble methods including weighted voting, stacking, and blending.
    """
    
    def __init__(self, output_dir: str = "ensemble_artifacts"):
        self.output_dir = output_dir
        self.base_models = {}
        self.weights = None
        self.meta_learner = None
        self.blend_train = None
        self.is_fitted = False
        os.makedirs(output_dir, exist_ok=True)
    
    def get_default_base_models(self) -> dict:
        """
        Returns default base models for ensemble.
        """
        return {
            'xgboost': XGBClassifier(
                eval_metric='logloss', random_state=42,
                n_estimators=200, max_depth=7, learning_rate=0.1
            ),
            'lightgbm': LGBMClassifier(
                random_state=42, verbose=-1,
                n_estimators=200, max_depth=7, learning_rate=0.1
            ),
            'random_forest': RandomForestClassifier(
                random_state=42, n_estimators=200, max_depth=15
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42, n_estimators=150, max_depth=5, learning_rate=0.1
            )
        }
    
    def set_base_models(self, models: dict):
        """
        Set custom base models.
        """
        self.base_models = models
    
    def _fit_base_models(self, X, y):
        """
        Fit all base models.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        fitted_models = {}
        for name, model in self.base_models.items():
            print(f"  Fitting {name}...")
            model.fit(X, y)
            fitted_models[name] = model
        
        self.base_models = fitted_models
        return fitted_models
    
    def _get_base_predictions(self, X, use_proba: bool = True) -> np.ndarray:
        """
        Get predictions from all base models.
        """
        X = np.asarray(X)
        predictions = []
        
        for name, model in self.base_models.items():
            if use_proba and hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    # =========================================================================
    # Weighted Voting Ensemble
    # =========================================================================
    
    def fit_weighted_voting(self, X_train, y_train, X_val, y_val, 
                            optimize_weights: bool = True) -> dict:
        """
        Fit weighted voting ensemble with optional weight optimization.
        """
        print("\n" + "="*50)
        print("Fitting Weighted Voting Ensemble")
        print("="*50)
        
        if not self.base_models:
            self.base_models = self.get_default_base_models()
        
        # Fit base models
        self._fit_base_models(X_train, y_train)
        
        # Get validation predictions
        val_preds = self._get_base_predictions(X_val, use_proba=True)
        y_val = np.asarray(y_val)
        
        if optimize_weights:
            print("\nOptimizing weights...")
            self.weights = self._optimize_weights(val_preds, y_val)
        else:
            # Equal weights
            n_models = len(self.base_models)
            self.weights = np.ones(n_models) / n_models
        
        # Evaluate
        weighted_proba = np.average(val_preds, axis=1, weights=self.weights)
        weighted_pred = (weighted_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, weighted_pred),
            'f1': f1_score(y_val, weighted_pred),
            'roc_auc': roc_auc_score(y_val, weighted_proba),
            'weights': dict(zip(self.base_models.keys(), self.weights.tolist()))
        }
        
        print(f"\nOptimized Weights: {metrics['weights']}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.is_fitted = True
        return metrics
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray, 
                          metric: str = 'f1') -> np.ndarray:
        """
        Optimize ensemble weights using scipy minimize.
        """
        n_models = predictions.shape[1]
        
        def objective(weights):
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / weights.sum()
            
            # Weighted average
            weighted_proba = np.average(predictions, axis=1, weights=weights)
            weighted_pred = (weighted_proba > 0.5).astype(int)
            
            # Maximize metric (minimize negative)
            if metric == 'f1':
                return -f1_score(y_true, weighted_pred)
            elif metric == 'accuracy':
                return -accuracy_score(y_true, weighted_pred)
            else:
                return -roc_auc_score(y_true, weighted_proba)
        
        # Initial equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Bounds: weights between 0 and 1
        bounds = [(0.01, 1.0)] * n_models
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(
            objective, initial_weights, 
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        
        # Normalize result
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def predict_weighted(self, X) -> np.ndarray:
        """
        Make predictions using weighted voting.
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit_weighted_voting first.")
        
        base_preds = self._get_base_predictions(X, use_proba=True)
        weighted_proba = np.average(base_preds, axis=1, weights=self.weights)
        return (weighted_proba > 0.5).astype(int)
    
    def predict_proba_weighted(self, X) -> np.ndarray:
        """
        Get probability predictions using weighted voting.
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit_weighted_voting first.")
        
        base_preds = self._get_base_predictions(X, use_proba=True)
        weighted_proba = np.average(base_preds, axis=1, weights=self.weights)
        return np.column_stack([1 - weighted_proba, weighted_proba])
    
    # =========================================================================
    # Stacking Ensemble
    # =========================================================================
    
    def fit_stacking(self, X_train, y_train, X_val, y_val,
                     meta_learner=None, use_features: bool = False) -> dict:
        """
        Fit stacking ensemble with meta-learner.
        """
        print("\n" + "="*50)
        print("Fitting Stacking Ensemble")
        print("="*50)
        
        if not self.base_models:
            self.base_models = self.get_default_base_models()
        
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        
        # Generate out-of-fold predictions for training meta-learner
        print("\nGenerating out-of-fold predictions...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        oof_predictions = np.zeros((len(y_train), len(self.base_models)))
        
        for idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  {name}...")
            oof_pred = cross_val_predict(
                model, X_train, y_train, cv=skf, method='predict_proba'
            )[:, 1]
            oof_predictions[:, idx] = oof_pred
        
        # Fit base models on full training data
        print("\nFitting base models on full training data...")
        self._fit_base_models(X_train, y_train)
        
        # Prepare meta-features
        if use_features:
            meta_train = np.hstack([oof_predictions, X_train])
        else:
            meta_train = oof_predictions
        
        # Fit meta-learner
        print("Fitting meta-learner...")
        self.meta_learner = meta_learner
        self.meta_learner.fit(meta_train, y_train)
        self.use_features = use_features
        
        # Evaluate on validation
        val_base_preds = self._get_base_predictions(X_val, use_proba=True)
        if use_features:
            meta_val = np.hstack([val_base_preds, X_val])
        else:
            meta_val = val_base_preds
        
        val_pred = self.meta_learner.predict(meta_val)
        val_proba = self.meta_learner.predict_proba(meta_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
            'roc_auc': roc_auc_score(y_val, val_proba)
        }
        
        print(f"\nValidation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.is_fitted = True
        return metrics
    
    def predict_stacking(self, X) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        """
        if self.meta_learner is None:
            raise ValueError("Model not fitted. Call fit_stacking first.")
        
        X = np.asarray(X)
        base_preds = self._get_base_predictions(X, use_proba=True)
        
        if self.use_features:
            meta_features = np.hstack([base_preds, X])
        else:
            meta_features = base_preds
        
        return self.meta_learner.predict(meta_features)
    
    def predict_proba_stacking(self, X) -> np.ndarray:
        """
        Get probability predictions using stacking.
        """
        if self.meta_learner is None:
            raise ValueError("Model not fitted. Call fit_stacking first.")
        
        X = np.asarray(X)
        base_preds = self._get_base_predictions(X, use_proba=True)
        
        if self.use_features:
            meta_features = np.hstack([base_preds, X])
        else:
            meta_features = base_preds
        
        return self.meta_learner.predict_proba(meta_features)
    
    # =========================================================================
    # Blending Ensemble
    # =========================================================================
    
    def fit_blending(self, X_train, y_train, X_blend, y_blend, X_val, y_val,
                     meta_learner=None) -> dict:
        """
        Fit blending ensemble using holdout blend set.
        """
        print("\n" + "="*50)
        print("Fitting Blending Ensemble")
        print("="*50)
        
        if not self.base_models:
            self.base_models = self.get_default_base_models()
        
        if meta_learner is None:
            meta_learner = XGBClassifier(
                eval_metric='logloss', random_state=42,
                n_estimators=100, max_depth=3
            )
        
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_blend = np.asarray(X_blend)
        y_blend = np.asarray(y_blend)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        
        # Fit base models on training data
        print("\nFitting base models...")
        self._fit_base_models(X_train, y_train)
        
        # Generate blend features
        print("Generating blend features...")
        blend_preds = self._get_base_predictions(X_blend, use_proba=True)
        
        # Fit meta-learner on blend set
        print("Fitting meta-learner...")
        self.meta_learner = meta_learner
        self.meta_learner.fit(blend_preds, y_blend)
        
        # Evaluate
        val_base_preds = self._get_base_predictions(X_val, use_proba=True)
        val_pred = self.meta_learner.predict(val_base_preds)
        val_proba = self.meta_learner.predict_proba(val_base_preds)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, val_pred),
            'f1': f1_score(y_val, val_pred),
            'roc_auc': roc_auc_score(y_val, val_proba)
        }
        
        print(f"\nValidation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")
        print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.use_features = False
        self.is_fitted = True
        return metrics
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def compare_ensembles(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        """
        Compare different ensemble methods.
        """
        print("\n" + "#"*60)
        print("Comparing Ensemble Methods")
        print("#"*60)
        
        results = []
        
        # Weighted Voting
        self.base_models = self.get_default_base_models()
        wv_metrics = self.fit_weighted_voting(X_train, y_train, X_val, y_val)
        results.append({
            'method': 'Weighted Voting',
            'accuracy': wv_metrics['accuracy'],
            'f1': wv_metrics['f1'],
            'roc_auc': wv_metrics['roc_auc']
        })
        
        # Stacking
        self.base_models = self.get_default_base_models()
        stack_metrics = self.fit_stacking(X_train, y_train, X_val, y_val)
        results.append({
            'method': 'Stacking',
            'accuracy': stack_metrics['accuracy'],
            'f1': stack_metrics['f1'],
            'roc_auc': stack_metrics['roc_auc']
        })
        
        # Stacking with features
        self.base_models = self.get_default_base_models()
        stack_feat_metrics = self.fit_stacking(
            X_train, y_train, X_val, y_val, use_features=True
        )
        results.append({
            'method': 'Stacking + Features',
            'accuracy': stack_feat_metrics['accuracy'],
            'f1': stack_feat_metrics['f1'],
            'roc_auc': stack_feat_metrics['roc_auc']
        })
        
        return pd.DataFrame(results).sort_values('f1', ascending=False)
    
    def save(self, filename: str = "advanced_ensemble.joblib"):
        """
        Save ensemble to file.
        """
        path = os.path.join(self.output_dir, filename)
        state = {
            'base_models': self.base_models,
            'weights': self.weights,
            'meta_learner': self.meta_learner,
            'use_features': getattr(self, 'use_features', False),
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, path)
        print(f"Saved ensemble to {path}")
    
    def load(self, filename: str = "advanced_ensemble.joblib"):
        """
        Load ensemble from file.
        """
        path = os.path.join(self.output_dir, filename)
        state = joblib.load(path)
        self.base_models = state['base_models']
        self.weights = state['weights']
        self.meta_learner = state['meta_learner']
        self.use_features = state.get('use_features', False)
        self.is_fitted = state['is_fitted']
        print(f"Loaded ensemble from {path}")
