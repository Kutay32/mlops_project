import logging
import os
import time
from datetime import datetime
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Import fallback mechanism
try:
    from src.serving.fallback import (
        ChurnFallbackPredictor,
        FallbackChain,
        FallbackMetrics,
    )

    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    logger.warning("Fallback module not available")

app = FastAPI(
    title="Churn Prediction Service",
    description="Production ML service with Algorithmic Fallback support",
    version="2.0.0",
)

# Enable Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app)

# Global variables
model_pipeline = None
secondary_model = None
model_loaded_at = None
prediction_count = 0
fallback_chain = None
fallback_metrics = None


class PredictionRequest(BaseModel):
    features: Dict[str, Union[str, float, int]]


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    model_used: Optional[str] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class FallbackStats(BaseModel):
    total_predictions: int
    fallback_count: int
    fallback_rate: float
    fallback_reasons: Dict[str, int]


class ModelInfo(BaseModel):
    model_loaded: bool
    model_path: Optional[str] = None
    loaded_at: Optional[str] = None
    predictions_made: int = 0


@app.on_event("startup")
async def load_model():
    global model_pipeline, secondary_model, model_loaded_at, fallback_chain, fallback_metrics

    # Look for model in multiple locations (Docker and local development)
    model_paths = [
        "model_artifact.joblib",
        "/app/model_artifact.joblib",
        "../model_artifact.joblib",
        "ultimate_churn_model.joblib",
        "/app/ultimate_churn_model.joblib",
    ]

    # Secondary model paths (simpler model for fallback)
    secondary_paths = [
        "model_artifacts/base_model_logistic_regression.joblib",
        "/app/model_artifacts/base_model_logistic_regression.joblib",
    ]

    # Load primary model
    for path in model_paths:
        if os.path.exists(path):
            try:
                model_pipeline = joblib.load(path)
                model_loaded_at = datetime.now().isoformat()
                logger.info(f"✓ Primary model loaded from {path}")
                break
            except Exception as e:
                logger.error(f"✗ Failed to load model from {path}: {e}")

    # Load secondary model for fallback
    for path in secondary_paths:
        if os.path.exists(path):
            try:
                secondary_model = joblib.load(path)
                logger.info(f"✓ Secondary model loaded from {path}")
                break
            except Exception as e:
                logger.warning(f"Could not load secondary model: {e}")

    # Initialize fallback chain
    if FALLBACK_AVAILABLE:
        fallback_predictor = ChurnFallbackPredictor(
            primary_model=model_pipeline, secondary_model=secondary_model
        )
        fallback_chain = fallback_predictor.create_fallback_chain()
        fallback_metrics = FallbackMetrics()
        logger.info("✓ Fallback chain initialized")

    if model_pipeline is None:
        logger.warning("⚠ Primary model not found. Using fallback only.")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    global prediction_count

    features = request.features

    # Try fallback chain first if available
    if FALLBACK_AVAILABLE and fallback_chain:
        try:
            result = fallback_chain.predict(features)
            prediction_count += 1

            # Record metrics
            if fallback_metrics:
                fallback_metrics.record_prediction(result)

            return PredictionResponse(
                churn_prediction=int(result.prediction),
                churn_probability=(
                    float(result.probability) if result.probability else 0.5
                ),
                model_used=result.model_used,
                is_fallback=result.is_fallback,
                fallback_reason=(
                    result.fallback_reason.value if result.fallback_reason else None
                ),
            )
        except Exception as e:
            logger.error(f"Fallback chain error: {e}")

    # Fallback to direct model prediction
    if not model_pipeline:
        # Ultimate fallback: business rules only
        if FALLBACK_AVAILABLE:
            predictor = ChurnFallbackPredictor()
            pred, prob = predictor.rules_based_predict(features)
            return PredictionResponse(
                churn_prediction=pred,
                churn_probability=prob,
                model_used="rules_based",
                is_fallback=True,
                fallback_reason="model_error",
            )
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Direct model prediction (legacy path)
    input_df = pd.DataFrame([features])

    try:
        start_time = time.time()
        prediction = model_pipeline.predict(input_df)[0]

        if hasattr(model_pipeline, "predict_proba"):
            probs = model_pipeline.predict_proba(input_df)
            probability = probs[0][1] if probs.shape[1] > 1 else probs[0][0]
        else:
            probability = float(prediction)

        prediction_count += 1
        inference_time = time.time() - start_time
        logger.info(
            f"Prediction #{prediction_count} completed in {inference_time:.4f}s"
        )

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=float(probability),
            model_used="primary_model",
            is_fallback=False,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")

        # Use fallback on error
        if FALLBACK_AVAILABLE:
            predictor = ChurnFallbackPredictor()
            pred, prob = predictor.rules_based_predict(features)
            return PredictionResponse(
                churn_prediction=pred,
                churn_probability=prob,
                model_used="rules_based",
                is_fallback=True,
                fallback_reason="model_error",
            )

        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/ready")
def readiness_check():
    """Kubernetes readiness probe endpoint."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_loaded=model_pipeline is not None,
        model_path="model_artifact.joblib" if model_pipeline else None,
        loaded_at=model_loaded_at,
        predictions_made=prediction_count,
    )


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "Churn Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics" if PROMETHEUS_AVAILABLE else "Not available",
        "fallback_enabled": FALLBACK_AVAILABLE,
    }


@app.get("/fallback-stats", response_model=FallbackStats)
def get_fallback_stats():
    """Get fallback usage statistics."""
    if not FALLBACK_AVAILABLE or not fallback_metrics:
        return FallbackStats(
            total_predictions=prediction_count,
            fallback_count=0,
            fallback_rate=0.0,
            fallback_reasons={},
        )

    summary = fallback_metrics.get_summary()
    return FallbackStats(
        total_predictions=summary["total_predictions"],
        fallback_count=summary["fallback_count"],
        fallback_rate=summary["fallback_rate"],
        fallback_reasons=summary["fallback_reasons"],
    )
