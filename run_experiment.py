"""
ML Optimization Experiment Script
Runs comprehensive model evaluation, hyperparameter tuning, and ensemble optimization.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.feature_engineering import HashingEncoder, EmbeddingEncoder
from src.models.model_trainer import ModelTrainer
from src.models.hyperparameter_tuner import HyperparameterTuner
from src.models.advanced_ensemble import AdvancedEnsemble
from src.evaluation.benchmarking import Benchmarker, DeploymentAnalyzer
from src.monitoring.production_monitor import create_monitor_for_production

warnings.filterwarnings("ignore")


def load_and_preprocess_data(data_path: str) -> tuple:
    """
    Load and preprocess the Telco Churn dataset.
    Returns preprocessed features and target.
    """
    print("\n" + "="*60)
    print("Loading and Preprocessing Data")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Basic cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop_duplicates()
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Handle target
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    df = df.dropna(subset=['Churn'])
    
    # Separate features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the feature preprocessor.
    """
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                            'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                            'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                            'Contract', 'PaymentMethod', 'InternetService']
    
    # Filter to existing columns
    numeric_features = [f for f in numeric_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
        ],
        remainder='drop'
    )
    
    return preprocessor


def run_experiment(data_path: str, run_tuning: bool = False):
    """
    Run the complete ML optimization experiment.
    """
    print("\n" + "#"*70)
    print("#" + " "*25 + "ML OPTIMIZATION EXPERIMENT" + " "*25 + "#")
    print("#"*70)
    
    # Load data
    X, y = load_and_preprocess_data(data_path)
    
    # Build preprocessor and transform
    print("\nBuilding preprocessor...")
    preprocessor = build_preprocessor(X)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")
    
    # Fit preprocessor and transform data
    print("\nTransforming features...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Convert to dense arrays
    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_val_transformed, 'toarray'):
        X_val_transformed = X_val_transformed.toarray()
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    
    print(f"Transformed feature dimension: {X_train_transformed.shape[1]}")
    
    # =========================================================================
    # Step 1: Train and evaluate individual models
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: Individual Model Training & Evaluation")
    print("="*70)
    
    trainer = ModelTrainer(output_dir="model_artifacts")
    
    # Train all models
    results_df = trainer.train_all_models(
        X_train_transformed, y_train,
        X_val_transformed, y_val
    )
    
    print("\n" + "-"*50)
    print("Individual Model Results (sorted by F1):")
    print("-"*50)
    print(results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    # Cross-validation
    print("\n" + "-"*50)
    print("Cross-Validation Results:")
    print("-"*50)
    cv_results = trainer.cross_validate_models(X_train_transformed, y_train, cv=5)
    print(cv_results.to_string(index=False))
    
    # Save individual models
    trainer.save_models()
    
    # =========================================================================
    # Step 2: Hyperparameter Tuning (optional - time consuming)
    # =========================================================================
    if run_tuning:
        print("\n" + "="*70)
        print("STEP 2: Hyperparameter Tuning")
        print("="*70)
        
        tuner = HyperparameterTuner(output_dir="tuning_results")
        
        # Tune top 2 models
        best_name, _, best_metrics = trainer.get_best_model('f1')
        print(f"\nTuning best model: {best_name}")
        
        tuning_df, tuned_estimators = tuner.tune_all_models(
            X_train_transformed, y_train, method='grid', cv=3
        )
        
        print("\n" + "-"*50)
        print("Tuning Results:")
        print("-"*50)
        print(tuning_df.to_string(index=False))
        
        tuner.save_results()
    else:
        print("\n[Skipping hyperparameter tuning - set run_tuning=True to enable]")
    
    # =========================================================================
    # Step 3: Ensemble Optimization
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: Ensemble Optimization")
    print("="*70)
    
    ensemble = AdvancedEnsemble(output_dir="ensemble_artifacts")
    
    # Weighted Voting
    print("\n--- Weighted Voting Ensemble ---")
    wv_metrics = ensemble.fit_weighted_voting(
        X_train_transformed, y_train,
        X_val_transformed, y_val,
        optimize_weights=True
    )
    
    # Evaluate on test set
    wv_test_pred = ensemble.predict_weighted(X_test_transformed)
    wv_test_proba = ensemble.predict_proba_weighted(X_test_transformed)[:, 1]
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    wv_test_metrics = {
        'accuracy': accuracy_score(y_test, wv_test_pred),
        'f1': f1_score(y_test, wv_test_pred),
        'roc_auc': roc_auc_score(y_test, wv_test_proba)
    }
    
    print(f"\nWeighted Voting - Test Metrics:")
    print(f"  Accuracy: {wv_test_metrics['accuracy']:.4f}")
    print(f"  F1: {wv_test_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {wv_test_metrics['roc_auc']:.4f}")
    
    # Stacking
    print("\n--- Stacking Ensemble ---")
    ensemble_stack = AdvancedEnsemble(output_dir="ensemble_artifacts")
    stack_metrics = ensemble_stack.fit_stacking(
        X_train_transformed, y_train,
        X_val_transformed, y_val,
        use_features=False
    )
    
    stack_test_pred = ensemble_stack.predict_stacking(X_test_transformed)
    stack_test_proba = ensemble_stack.predict_proba_stacking(X_test_transformed)[:, 1]
    
    stack_test_metrics = {
        'accuracy': accuracy_score(y_test, stack_test_pred),
        'f1': f1_score(y_test, stack_test_pred),
        'roc_auc': roc_auc_score(y_test, stack_test_proba)
    }
    
    print(f"\nStacking - Test Metrics:")
    print(f"  Accuracy: {stack_test_metrics['accuracy']:.4f}")
    print(f"  F1: {stack_test_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {stack_test_metrics['roc_auc']:.4f}")
    
    # Save best ensemble
    if wv_test_metrics['f1'] > stack_test_metrics['f1']:
        ensemble.save("best_ensemble.joblib")
        print("\nSaved Weighted Voting as best ensemble")
    else:
        ensemble_stack.save("best_ensemble.joblib")
        print("\nSaved Stacking as best ensemble")
    
    # =========================================================================
    # Step 4: Final Benchmarking
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: Final Benchmarking")
    print("="*70)
    
    benchmarker = Benchmarker(output_dir="benchmark_results")
    benchmarker.set_target_metric(0.7850)  # Current baseline
    
    # Benchmark individual models
    models = trainer.get_base_models()
    for name, model in models.items():
        benchmarker.benchmark_model(
            name, model, X_train_transformed, y_train,
            X_test_transformed, y_test, cv=5
        )
    
    # Add ensemble results
    benchmarker.results.append({
        'model_name': 'Ensemble (Weighted Voting)',
        'test_accuracy': wv_test_metrics['accuracy'],
        'test_precision': 0.0,  # Not computed
        'test_recall': 0.0,
        'test_f1': wv_test_metrics['f1'],
        'test_roc_auc': wv_test_metrics['roc_auc'],
        'train_time_seconds': 0,
        'inference_time_seconds': 0,
        'inference_time_per_sample_ms': 0,
        'beats_target': wv_test_metrics['accuracy'] > 0.7850
    })
    
    benchmarker.results.append({
        'model_name': 'Ensemble (Stacking)',
        'test_accuracy': stack_test_metrics['accuracy'],
        'test_precision': 0.0,
        'test_recall': 0.0,
        'test_f1': stack_test_metrics['f1'],
        'test_roc_auc': stack_test_metrics['roc_auc'],
        'train_time_seconds': 0,
        'inference_time_seconds': 0,
        'inference_time_per_sample_ms': 0,
        'beats_target': stack_test_metrics['accuracy'] > 0.7850
    })
    
    # Create results dataframe
    benchmark_df = pd.DataFrame(benchmarker.results)
    benchmark_df = benchmark_df.sort_values('test_accuracy', ascending=False)
    
    # Print summary
    benchmarker.print_summary(benchmark_df)
    
    # Save results
    benchmarker.save_results(benchmark_df)
    
    # =========================================================================
    # Step 5: Deployment Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: Deployment Analysis")
    print("="*70)
    
    analyzer = DeploymentAnalyzer()
    
    # Latency analysis
    trained_models = {}
    for name, model in trainer.trained_models.items():
        trained_models[name] = model
    
    if trained_models:
        latency_df = analyzer.analyze_latency(trained_models, X_test_transformed, n_runs=50)
        memory_df = analyzer.analyze_memory(trained_models)
    
    # Generate deployment report
    report = analyzer.generate_deployment_report(benchmark_df)
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "#"*70)
    print("#" + " "*25 + "EXPERIMENT COMPLETE" + " "*25 + "#")
    print("#"*70)
    
    best = benchmark_df.iloc[0]
    print(f"\nBest Performing Model: {best['model_name']}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Test F1 Score: {best['test_f1']:.4f}")
    
    if best['test_accuracy'] > 0.7850:
        improvement = (best['test_accuracy'] - 0.7850) * 100
        print(f"\n✓ SUCCESS: Exceeded target by {improvement:.2f}%")
    else:
        gap = (0.7850 - best['test_accuracy']) * 100
        print(f"\n✗ Target not met. Gap: {gap:.2f}%")
    
    print("\nArtifacts saved:")
    print("  - model_artifacts/     : Trained individual models")
    print("  - ensemble_artifacts/  : Ensemble models")
    print("  - benchmark_results/   : Performance comparisons")
    
    return benchmark_df


if __name__ == "__main__":
    # Data URL
    data_url = "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
    
    # Run experiment
    try:
        results = run_experiment(data_url, run_tuning=False)
    except Exception as e:
        print(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()
