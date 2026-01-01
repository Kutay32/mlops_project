from unittest.mock import MagicMock

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.models.model_trainer import ModelTrainer


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, n_redundant=0, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_model_trainer_mlflow_logging(sample_data, tmp_path):
    X_train, X_test, y_train, y_test = sample_data

    trainer = ModelTrainer(output_dir=str(tmp_path), use_mlflow=False)

    # Inject fake MLflow client and sklearn logger
    trainer.use_mlflow = True

    fake_mlflow = MagicMock()
    fake_context = MagicMock()
    fake_context.__enter__.return_value = fake_context
    fake_context.__exit__.return_value = False
    fake_mlflow.start_run.return_value = fake_context
    fake_mlflow.log_params = MagicMock()
    fake_mlflow.log_metric = MagicMock()
    fake_mlflow.log_artifact = MagicMock()
    fake_mlflow.set_tag = MagicMock()

    fake_mlflow_sklearn = MagicMock()
    fake_mlflow_sklearn.log_model = MagicMock()

    trainer.mlflow = fake_mlflow
    trainer.mlflow_sklearn = fake_mlflow_sklearn

    # Train a logistic regression
    metrics = trainer.train_model(
        "logistic_regression",
        LogisticRegression(max_iter=1000, random_state=42),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Assertions: MLflow should have been invoked to log params and model
    assert fake_mlflow.start_run.called
    assert fake_mlflow.log_params.called
    assert fake_mlflow.log_metric.called
    assert fake_mlflow_sklearn.log_model.called
    # Verify that metrics were returned
    assert metrics is not None
