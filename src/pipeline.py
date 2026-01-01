import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from src.data.feature_engineering import HashingEncoder, EmbeddingEncoder
from src.models.ensemble import EnsembleModel
import joblib
import os


def _to_numpy(X):
    """Convert to plain numpy array to strip feature names."""
    # Create a fresh copy to completely strip any sklearn feature name metadata
    arr = np.array(X)
    return arr.copy()

class ChurnPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.pipeline = None

    def load_data(self):
        print(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Basic cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.drop_duplicates()
        
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
            
        # Handle target
        if 'Churn' in df.columns:
            # Drop rows with missing target
            df = df.dropna(subset=['Churn'])
            # Map Churn if needed
            if df['Churn'].dtype == 'object':
                 df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        return df

    def build_pipeline(self):
        # Define features based on Telco Churn dataset
        # Note: Columns must exist in the dataset
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Standard categorical features for OneHot
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                'PhoneService', 'MultipleLines', 'OnlineSecurity', 
                                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
        
        # Demo Hashing on 'PaymentMethod' (Pattern 1)
        hashing_features = ['PaymentMethod']
        
        # Demo Embedding on 'Contract' (Pattern 2)
        # We apply separately to handle each column's embedding
        
        # Preprocessing steps
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
                
                # Apply HashingEncoder to PaymentMethod
                ('hash', HashingEncoder(n_features=8, col='PaymentMethod'), hashing_features),
                
                # Apply EmbeddingEncoder to Contract
                ('emb_contract', EmbeddingEncoder(embedding_dim=4, col='Contract'), ['Contract']),
                ('emb_internet', EmbeddingEncoder(embedding_dim=4, col='InternetService'), ['InternetService']),
            ],
            remainder='drop'
        )
        
        # Full Pipeline with Ensemble Model (Pattern 5) and Rebalancing (Pattern 6)
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('to_numpy', FunctionTransformer(_to_numpy)),
            ('classifier', EnsembleModel(ensemble_type='voting', rebalancing_strategy='SMOTE'))
        ])
        
        print("Pipeline built successfully.")

    def train(self, df):
        if 'Churn' not in df.columns:
            raise ValueError("Dataframe must contain 'Churn' column")
            
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        print(f"Training on {X_train.shape[0]} samples...")
        self.pipeline.fit(X_train, y_train)
        print("Training completed.")
        
        print("Evaluating...")
        score = self.pipeline.score(X_test, y_test)
        print(f"Test Accuracy: {score:.4f}")
        
        return score

    def save(self, path):
        joblib.dump(self.pipeline, path)
        print(f"Pipeline saved to {path}")

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
    
    try:
        pipeline = ChurnPipeline(url)
        df = pipeline.load_data()
        pipeline.build_pipeline()
        pipeline.train(df)
        pipeline.save("model_artifact.joblib")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
