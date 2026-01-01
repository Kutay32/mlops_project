"""
Hyperparameter Tuning Module
Implements grid search, random search, and Bayesian optimization for model tuning.
"""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class HyperparameterTuner:
    """
    Hyperparameter tuning for base models using various search strategies.
    """
    
    def __init__(self, output_dir: str = "tuning_results"):
        self.output_dir = output_dir
        self.best_params = {}
        self.tuning_history = []
        os.makedirs(output_dir, exist_ok=True)
    
    def get_param_grids(self) -> dict:
        """
        Returns parameter grids for each model type.
        """
        return {
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [15, 31, 63, 127],
                'min_child_samples': [10, 20, 30, 50],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.1, 0.5, 1.0, 2.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'max_features': ['sqrt', 'log2', None]
            }
        }
    
    def get_reduced_param_grids(self) -> dict:
        """
        Returns reduced parameter grids for faster tuning.
        """
        return {
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [5, 7],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_lambda': [0.5, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200],
                'max_depth': [5, 7],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [1, 2],
                'class_weight': [None, 'balanced']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        }
    
    def get_base_model(self, model_name: str):
        """
        Returns base model instance.
        """
        models = {
            'xgboost': XGBClassifier(eval_metric='logloss', random_state=42),
            'lightgbm': LGBMClassifier(random_state=42, verbose=-1),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        return models.get(model_name)
    
    def grid_search(self, model_name: str, X, y, param_grid: dict = None,
                    cv: int = 5, scoring: str = 'f1', n_jobs: int = -1) -> dict:
        """
        Perform grid search for a specific model.
        """
        print(f"\n{'='*50}")
        print(f"Grid Search: {model_name}")
        print(f"{'='*50}")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        model = self.get_base_model(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        if param_grid is None:
            param_grid = self.get_reduced_param_grids().get(model_name, {})
        
        # Calculate total combinations
        total_combinations = 1
        for v in param_grid.values():
            total_combinations *= len(v)
        print(f"Total parameter combinations: {total_combinations}")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        start_time = datetime.now()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=skf,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'tuning_time_seconds': tuning_time,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': grid_search.cv_results_['std_test_score'].tolist(),
                'mean_train_score': grid_search.cv_results_['mean_train_score'].tolist(),
            }
        }
        
        self.best_params[model_name] = grid_search.best_params_
        self.tuning_history.append(result)
        
        print(f"\nBest {scoring}: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        print(f"Tuning time: {tuning_time:.2f}s")
        
        return result, grid_search.best_estimator_
    
    def random_search(self, model_name: str, X, y, param_distributions: dict = None,
                      n_iter: int = 50, cv: int = 5, scoring: str = 'f1', 
                      n_jobs: int = -1) -> dict:
        """
        Perform randomized search for a specific model.
        """
        print(f"\n{'='*50}")
        print(f"Random Search: {model_name} (n_iter={n_iter})")
        print(f"{'='*50}")
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        model = self.get_base_model(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        if param_distributions is None:
            param_distributions = self.get_param_grids().get(model_name, {})
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        start_time = datetime.now()
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=skf,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        random_search.fit(X, y)
        
        tuning_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'model_name': model_name,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'tuning_time_seconds': tuning_time,
            'n_iter': n_iter
        }
        
        self.best_params[model_name] = random_search.best_params_
        self.tuning_history.append(result)
        
        print(f"\nBest {scoring}: {random_search.best_score_:.4f}")
        print(f"Best params: {random_search.best_params_}")
        print(f"Tuning time: {tuning_time:.2f}s")
        
        return result, random_search.best_estimator_
    
    def tune_all_models(self, X, y, method: str = 'grid', 
                        cv: int = 5, n_iter: int = 30) -> pd.DataFrame:
        """
        Tune all supported models.
        """
        print(f"\n{'#'*60}")
        print(f"Tuning All Models ({method} search)")
        print(f"{'#'*60}")
        
        results = []
        best_estimators = {}
        
        model_names = ['xgboost', 'lightgbm', 'random_forest', 'gradient_boosting']
        
        for name in model_names:
            try:
                if method == 'grid':
                    result, estimator = self.grid_search(name, X, y, cv=cv)
                else:
                    result, estimator = self.random_search(name, X, y, n_iter=n_iter, cv=cv)
                
                results.append({
                    'model_name': name,
                    'best_score': result['best_score'],
                    'tuning_time': result['tuning_time_seconds']
                })
                best_estimators[name] = estimator
                
            except Exception as e:
                print(f"Error tuning {name}: {e}")
                continue
        
        return pd.DataFrame(results).sort_values('best_score', ascending=False), best_estimators
    
    def save_results(self, filename: str = "tuning_results.json"):
        """
        Save tuning results to file.
        """
        path = os.path.join(self.output_dir, filename)
        
        # Convert numpy types
        serializable = []
        for result in self.tuning_history:
            sr = {}
            for k, v in result.items():
                if isinstance(v, np.floating):
                    sr[k] = float(v)
                elif isinstance(v, np.integer):
                    sr[k] = int(v)
                elif isinstance(v, dict):
                    sr[k] = {kk: float(vv) if isinstance(vv, np.floating) else vv 
                             for kk, vv in v.items()}
                else:
                    sr[k] = v
            serializable.append(sr)
        
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"Saved tuning results to {path}")
    
    def get_tuned_model(self, model_name: str):
        """
        Returns a model with best found parameters.
        """
        if model_name not in self.best_params:
            raise ValueError(f"No tuning results for {model_name}")
        
        model = self.get_base_model(model_name)
        model.set_params(**self.best_params[model_name])
        return model
