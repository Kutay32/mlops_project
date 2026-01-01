I will implement the ML pipeline following the requested OOP patterns and clean coding standards.

### 1. Project Structure Setup
- Create directory structure:
  - `src/`
    - `data/` (Feature engineering, loading)
    - `models/` (Model definitions, reframing, ensembles)
    - `serving/` (API endpoint)
- Create `requirements.txt` with necessary dependencies (`pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `fastapi`, `uvicorn`, `imbalanced-learn`).

### 2. Data Representation (src/data/)
- **`src/data/feature_engineering.py`**:
  - Implement `HashingEncoder` class using `sklearn.feature_extraction.FeatureHasher`.
  - Implement `EmbeddingEncoder` class using `numpy` for Xavier initialization and lookup.
  - Implement feature crossing logic (e.g., as a custom Transformer).

### 3. Problem Representation (src/models/)
- **`src/models/reframing.py`**:
  - Implement `RegressionToClassification` using `sklearn.preprocessing.KBinsDiscretizer`.
- **`src/models/base_model.py`**:
  - Create a base class handling common logic.
  - Implement `_apply_rebalancing` method supporting SMOTE, ADASYN, Undersampling, and Class Weights.
- **`src/models/ensemble.py`**:
  - Implement `EnsembleModel` class supporting Voting, Stacking, and Bagging.
  - Integrate `XGBoost`, `LightGBM`, and `RandomForest` as base learners.

### 4. Resilient Serving (src/serving/)
- **`src/serving/api.py`**:
  - Implement `FastAPI` application.
  - Define `PredictionRequest` and `PredictionResponse` schemas.
  - Create the `/predict` endpoint (stateless).

### 5. Orchestration & Training
- **`src/pipeline.py`**:
  - Migrate data loading/cleaning logic from `dataset.py` into a cleaner `DataLoader` class.
  - Create a training pipeline that:
    1. Loads data.
    2. Applies feature engineering (Hashing/Embeddings).
    3. Balances the dataset.
    4. Trains the Ensemble model.
    5. Saves the artifact for serving.
- **`main.py`**: Entry point to run the training pipeline.

### 6. Verification
- Run `main.py` to ensure the pipeline trains successfully.
- Start the API (dry run or background) to verify `src/serving/api.py` loads correctly.
