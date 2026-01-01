"""
Benchmarking Framework
Comprehensive comparison of individual models vs ensemble performance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer
)
from datetime import datetime
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")


class Benchmarker:
    """
    Benchmarks individual models against ensemble methods.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.target_metric = 0.7850  # Baseline to beat
        os.makedirs(output_dir, exist_ok=True)
    
    def set_target_metric(self, target: float):
        """
        Set the target metric to beat.
        """
        self.target_metric = target
        print(f"Target metric set to: {target:.4f}")
    
    def _compute_all_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """
        Compute all classification metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except:
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    def benchmark_model(self, name: str, model, X_train, y_train, 
                        X_test, y_test, cv: int = 5) -> dict:
        """
        Benchmark a single model with cross-validation.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get probabilities
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = self._compute_all_metrics(y_test, y_pred, y_proba)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=['accuracy', 'f1', 'roc_auc'],
            return_train_score=True
        )
        
        result = {
            'model_name': name,
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'test_roc_auc': metrics.get('roc_auc', 0.0),
            'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1_mean': cv_scores['test_f1'].mean(),
            'cv_f1_std': cv_scores['test_f1'].std(),
            'cv_roc_auc_mean': cv_scores['test_roc_auc'].mean(),
            'cv_roc_auc_std': cv_scores['test_roc_auc'].std(),
            'train_time_seconds': train_time,
            'inference_time_seconds': inference_time,
            'inference_time_per_sample_ms': (inference_time / len(X_test)) * 1000,
            'beats_target': metrics['accuracy'] > self.target_metric
        }
        
        self.results.append(result)
        return result
    
    def benchmark_ensemble(self, name: str, ensemble, X_train, y_train,
                           X_val, y_val, X_test, y_test, 
                           ensemble_type: str = 'weighted') -> dict:
        """
        Benchmark an ensemble method.
        """
        from src.models.advanced_ensemble import AdvancedEnsemble
        
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)
        
        # Training time
        start_time = time.time()
        
        if ensemble_type == 'weighted':
            ensemble.fit_weighted_voting(X_train, y_train, X_val, y_val)
        elif ensemble_type == 'stacking':
            ensemble.fit_stacking(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        train_time = time.time() - start_time
        
        # Inference time
        start_time = time.time()
        if ensemble_type == 'weighted':
            y_pred = ensemble.predict_weighted(X_test)
            y_proba = ensemble.predict_proba_weighted(X_test)[:, 1]
        else:
            y_pred = ensemble.predict_stacking(X_test)
            y_proba = ensemble.predict_proba_stacking(X_test)[:, 1]
        
        inference_time = time.time() - start_time
        
        # Compute metrics
        metrics = self._compute_all_metrics(y_test, y_pred, y_proba)
        
        result = {
            'model_name': name,
            'test_accuracy': metrics['accuracy'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall'],
            'test_f1': metrics['f1'],
            'test_roc_auc': metrics.get('roc_auc', 0.0),
            'train_time_seconds': train_time,
            'inference_time_seconds': inference_time,
            'inference_time_per_sample_ms': (inference_time / len(X_test)) * 1000,
            'beats_target': metrics['accuracy'] > self.target_metric
        }
        
        self.results.append(result)
        return result
    
    def run_full_benchmark(self, X, y, test_size: float = 0.2, 
                           val_size: float = 0.1) -> pd.DataFrame:
        """
        Run comprehensive benchmark comparing all models and ensembles.
        """
        from src.models.model_trainer import ModelTrainer
        from src.models.advanced_ensemble import AdvancedEnsemble
        
        print("\n" + "#"*60)
        print("Running Full Benchmark")
        print(f"Target Metric (Accuracy): {self.target_metric:.4f}")
        print("#"*60)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            stratify=y_temp, random_state=42
        )
        
        print(f"\nData splits:")
        print(f"  Train: {len(X_train)}")
        print(f"  Val:   {len(X_val)}")
        print(f"  Test:  {len(X_test)}")
        
        # Benchmark individual models
        print("\n" + "="*50)
        print("Benchmarking Individual Models")
        print("="*50)
        
        trainer = ModelTrainer()
        models = trainer.get_base_models()
        
        for name, model in models.items():
            print(f"\n{name}...")
            try:
                self.benchmark_model(name, model, X_train, y_train, X_test, y_test)
            except Exception as e:
                print(f"  Error: {e}")
        
        # Benchmark ensembles
        print("\n" + "="*50)
        print("Benchmarking Ensemble Methods")
        print("="*50)
        
        # Weighted Voting
        print("\nWeighted Voting Ensemble...")
        ensemble_wv = AdvancedEnsemble()
        try:
            self.benchmark_ensemble(
                "Ensemble (Weighted Voting)", ensemble_wv,
                X_train, y_train, X_val, y_val, X_test, y_test,
                ensemble_type='weighted'
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        # Stacking
        print("\nStacking Ensemble...")
        ensemble_stack = AdvancedEnsemble()
        try:
            self.benchmark_ensemble(
                "Ensemble (Stacking)", ensemble_stack,
                X_train, y_train, X_val, y_val, X_test, y_test,
                ensemble_type='stacking'
            )
        except Exception as e:
            print(f"  Error: {e}")
        
        # Create results DataFrame
        df = pd.DataFrame(self.results)
        df = df.sort_values('test_accuracy', ascending=False)
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """
        Print formatted benchmark summary.
        """
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Target Metric (Accuracy): {self.target_metric:.4f}")
        print("-"*80)
        
        # Key metrics
        cols = ['model_name', 'test_accuracy', 'test_f1', 'test_roc_auc', 
                'train_time_seconds', 'beats_target']
        
        if all(c in df.columns for c in cols):
            summary = df[cols].copy()
            summary.columns = ['Model', 'Accuracy', 'F1', 'ROC-AUC', 'Train Time(s)', 'Beats Target']
            print(summary.to_string(index=False))
        else:
            print(df.to_string(index=False))
        
        print("-"*80)
        
        # Best model
        best = df.iloc[0]
        print(f"\nBest Model: {best['model_name']}")
        print(f"  Accuracy: {best['test_accuracy']:.4f}")
        print(f"  F1 Score: {best['test_f1']:.4f}")
        
        if best['test_accuracy'] > self.target_metric:
            improvement = (best['test_accuracy'] - self.target_metric) * 100
            print(f"\n✓ Target EXCEEDED by {improvement:.2f}%")
        else:
            gap = (self.target_metric - best['test_accuracy']) * 100
            print(f"\n✗ Target NOT met. Gap: {gap:.2f}%")
    
    def save_results(self, df: pd.DataFrame, filename: str = "benchmark_results"):
        """
        Save benchmark results.
        """
        # CSV
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
        
        # JSON with metadata
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'target_metric': self.target_metric,
            'best_model': df.iloc[0]['model_name'],
            'best_accuracy': float(df.iloc[0]['test_accuracy']),
            'beats_target': bool(df.iloc[0]['test_accuracy'] > self.target_metric),
            'results': df.to_dict(orient='records')
        }
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        results_dict = convert_types(results_dict)
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Saved detailed results to {json_path}")


class DeploymentAnalyzer:
    """
    Analyzes deployment considerations for models.
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_latency(self, models: dict, X_test, n_runs: int = 100) -> pd.DataFrame:
        """
        Analyze inference latency for each model.
        """
        print("\n" + "="*50)
        print("Latency Analysis")
        print("="*50)
        
        X_test = np.asarray(X_test)
        results = []
        
        for name, model in models.items():
            latencies = []
            
            for _ in range(n_runs):
                # Single sample inference
                sample = X_test[0:1]
                start = time.time()
                model.predict(sample)
                latencies.append((time.time() - start) * 1000)  # ms
            
            # Batch inference
            start = time.time()
            model.predict(X_test)
            batch_time = (time.time() - start) * 1000
            
            results.append({
                'model_name': name,
                'single_sample_mean_ms': np.mean(latencies),
                'single_sample_p50_ms': np.percentile(latencies, 50),
                'single_sample_p95_ms': np.percentile(latencies, 95),
                'single_sample_p99_ms': np.percentile(latencies, 99),
                'batch_total_ms': batch_time,
                'batch_per_sample_ms': batch_time / len(X_test)
            })
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        self.analysis_results['latency'] = df
        return df
    
    def analyze_memory(self, models: dict) -> pd.DataFrame:
        """
        Estimate memory requirements for each model.
        """
        import sys
        import pickle
        
        print("\n" + "="*50)
        print("Memory Analysis")
        print("="*50)
        
        results = []
        
        for name, model in models.items():
            # Serialize to estimate size
            try:
                serialized = pickle.dumps(model)
                size_mb = len(serialized) / (1024 * 1024)
            except:
                size_mb = 0.0
            
            results.append({
                'model_name': name,
                'serialized_size_mb': size_mb
            })
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        self.analysis_results['memory'] = df
        return df
    
    def generate_deployment_report(self, benchmark_df: pd.DataFrame) -> dict:
        """
        Generate comprehensive deployment report.
        """
        print("\n" + "="*60)
        print("DEPLOYMENT ANALYSIS REPORT")
        print("="*60)
        
        best_model = benchmark_df.iloc[0]
        
        report = {
            'recommended_model': best_model['model_name'],
            'performance': {
                'accuracy': float(best_model['test_accuracy']),
                'f1': float(best_model['test_f1']),
                'roc_auc': float(best_model.get('test_roc_auc', 0))
            },
            'computational': {
                'train_time_seconds': float(best_model['train_time_seconds']),
                'inference_time_per_sample_ms': float(best_model['inference_time_per_sample_ms'])
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if best_model['inference_time_per_sample_ms'] < 1:
            report['recommendations'].append(
                "✓ Low latency - suitable for real-time inference"
            )
        elif best_model['inference_time_per_sample_ms'] < 10:
            report['recommendations'].append(
                "⚠ Moderate latency - consider caching for high-traffic scenarios"
            )
        else:
            report['recommendations'].append(
                "✗ High latency - consider model distillation or simpler models"
            )
        
        if 'Ensemble' in best_model['model_name']:
            report['recommendations'].append(
                "⚠ Ensemble model - requires multiple model inference calls"
            )
            report['recommendations'].append(
                "Consider: Model distillation to single model for production"
            )
        
        # Print report
        print(f"\nRecommended Model: {report['recommended_model']}")
        print(f"\nPerformance:")
        for k, v in report['performance'].items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\nComputational Requirements:")
        for k, v in report['computational'].items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        return report
