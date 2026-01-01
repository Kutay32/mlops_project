"""
Algorithmic Fallback Pattern Implementation
Provides backup prediction mechanisms when primary model fails
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for triggering fallback"""
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    LOW_CONFIDENCE = "low_confidence"
    DRIFT_DETECTED = "drift_detected"
    RESOURCE_EXHAUSTED = "resource_exhausted"


@dataclass
class PredictionResult:
    """Result container for predictions with metadata"""
    prediction: Any
    probability: Optional[float]
    model_used: str
    is_fallback: bool
    fallback_reason: Optional[FallbackReason]
    latency_ms: float
    timestamp: datetime
    metadata: Dict


class FallbackChain:
    """
    Implements the Algorithmic Fallback Pattern.
    
    Creates a chain of prediction methods, falling back to simpler
    approaches when more complex methods fail.
    
    Fallback Hierarchy:
    1. Primary ML Model (most complex/accurate)
    2. Secondary ML Model (simpler/faster)
    3. Rules-Based System (deterministic)
    4. Default Value (constant)
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        timeout_seconds: float = 5.0,
        max_retries: int = 2
    ):
        """
        Initialize fallback chain.
        
        Args:
            confidence_threshold: Minimum confidence for accepting prediction
            timeout_seconds: Timeout for each prediction attempt
            max_retries: Maximum retries before falling back
        """
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        self.fallback_handlers: List[Tuple[str, Callable]] = []
        self.fallback_counts: Dict[str, int] = {}
        self.total_predictions = 0
    
    def add_handler(self, name: str, handler: Callable):
        """
        Add a fallback handler to the chain.
        
        Args:
            name: Handler name for logging
            handler: Callable that takes features and returns (prediction, probability)
        """
        self.fallback_handlers.append((name, handler))
        self.fallback_counts[name] = 0
        logger.info(f"Added fallback handler: {name}")
    
    def predict(self, features: Any) -> PredictionResult:
        """
        Make prediction with fallback support.
        
        Args:
            features: Input features for prediction
            
        Returns:
            PredictionResult with prediction and metadata
        """
        start_time = datetime.now()
        self.total_predictions += 1
        
        for i, (name, handler) in enumerate(self.fallback_handlers):
            is_fallback = i > 0
            
            try:
                prediction, probability = handler(features)
                
                # Check confidence threshold
                if probability is not None and probability < self.confidence_threshold:
                    logger.warning(
                        f"Handler {name} returned low confidence ({probability:.3f}), "
                        f"trying next fallback"
                    )
                    continue
                
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.fallback_counts[name] += 1
                
                result = PredictionResult(
                    prediction=prediction,
                    probability=probability,
                    model_used=name,
                    is_fallback=is_fallback,
                    fallback_reason=FallbackReason.LOW_CONFIDENCE if is_fallback else None,
                    latency_ms=latency,
                    timestamp=datetime.now(),
                    metadata={
                        "handler_index": i,
                        "total_handlers": len(self.fallback_handlers)
                    }
                )
                
                if is_fallback:
                    logger.warning(f"Used fallback handler: {name}")
                
                return result
                
            except Exception as e:
                logger.error(f"Handler {name} failed: {e}")
                continue
        
        # All handlers failed - use ultimate fallback
        latency = (datetime.now() - start_time).total_seconds() * 1000
        logger.error("All handlers failed, using ultimate fallback")
        
        return PredictionResult(
            prediction=0,  # Default: no churn
            probability=0.5,  # Neutral probability
            model_used="ultimate_fallback",
            is_fallback=True,
            fallback_reason=FallbackReason.MODEL_ERROR,
            latency_ms=latency,
            timestamp=datetime.now(),
            metadata={"error": "all_handlers_failed"}
        )
    
    def get_stats(self) -> Dict:
        """Get fallback statistics"""
        return {
            "total_predictions": self.total_predictions,
            "handler_usage": self.fallback_counts,
            "fallback_rate": (
                sum(v for k, v in self.fallback_counts.items() if k != self.fallback_handlers[0][0])
                / max(self.total_predictions, 1)
            )
        }


class ChurnFallbackPredictor:
    """
    Churn-specific fallback predictor with business rules.
    
    Implements a 4-tier fallback hierarchy:
    1. Primary model (e.g., XGBoost)
    2. Simple model (e.g., Logistic Regression)
    3. Business rules
    4. Base rate
    """
    
    def __init__(self, primary_model=None, secondary_model=None):
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.base_churn_rate = 0.27  # Historical churn rate
        
        # Business rules thresholds
        self.high_risk_tenure_threshold = 6  # months
        self.high_risk_contract = "Month-to-month"
        self.high_risk_charges_threshold = 70  # monthly
    
    def rules_based_predict(self, features: Dict) -> Tuple[int, float]:
        """
        Rules-based prediction using business logic.
        
        High risk indicators:
        - Short tenure (< 6 months)
        - Month-to-month contract
        - High monthly charges
        - No tech support
        - Fiber optic with no security
        """
        risk_score = 0.0
        
        # Tenure risk
        tenure = features.get('tenure', 12)
        if tenure < self.high_risk_tenure_threshold:
            risk_score += 0.25
        
        # Contract risk
        contract = features.get('Contract', 'Two year')
        if contract == self.high_risk_contract:
            risk_score += 0.20
        
        # Charges risk
        monthly_charges = features.get('MonthlyCharges', 50)
        if monthly_charges > self.high_risk_charges_threshold:
            risk_score += 0.15
        
        # Service risk
        if features.get('TechSupport') == 'No':
            risk_score += 0.10
        
        if features.get('InternetService') == 'Fiber optic':
            if features.get('OnlineSecurity') == 'No':
                risk_score += 0.15
        
        # Billing risk
        if features.get('PaymentMethod') == 'Electronic check':
            risk_score += 0.10
        
        # Convert to prediction
        probability = min(risk_score + self.base_churn_rate, 0.95)
        prediction = 1 if probability > 0.5 else 0
        
        return prediction, probability
    
    def base_rate_predict(self, features: Dict) -> Tuple[int, float]:
        """
        Ultimate fallback: predict based on historical base rate.
        Returns: (prediction, base_rate_probability)
        """
        return 0, self.base_churn_rate
    
    def create_fallback_chain(self) -> FallbackChain:
        """Create configured fallback chain"""
        chain = FallbackChain(confidence_threshold=0.3)
        
        # Level 1: Primary model
        if self.primary_model:
            def primary_predict(features):
                X = self._prepare_features(features)
                pred = self.primary_model.predict(X)[0]
                proba = self.primary_model.predict_proba(X)[0, 1]
                return int(pred), float(proba)
            chain.add_handler("primary_model", primary_predict)
        
        # Level 2: Secondary model
        if self.secondary_model:
            def secondary_predict(features):
                X = self._prepare_features(features)
                pred = self.secondary_model.predict(X)[0]
                proba = self.secondary_model.predict_proba(X)[0, 1]
                return int(pred), float(proba)
            chain.add_handler("secondary_model", secondary_predict)
        
        # Level 3: Business rules
        chain.add_handler("rules_based", self.rules_based_predict)
        
        # Level 4: Base rate
        chain.add_handler("base_rate", self.base_rate_predict)
        
        return chain
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Prepare features for model prediction"""
        # This should match your preprocessing pipeline
        # Simplified version - actual implementation depends on your feature engineering
        import pandas as pd
        return pd.DataFrame([features])


class FallbackMetrics:
    """Track and report fallback usage metrics"""
    
    def __init__(self):
        self.predictions = []
        self.fallback_events = []
    
    def record_prediction(self, result: PredictionResult):
        """Record a prediction result"""
        self.predictions.append({
            'timestamp': result.timestamp,
            'model_used': result.model_used,
            'is_fallback': result.is_fallback,
            'reason': result.fallback_reason.value if result.fallback_reason else None,
            'latency_ms': result.latency_ms
        })
        
        if result.is_fallback:
            self.fallback_events.append({
                'timestamp': result.timestamp,
                'model_used': result.model_used,
                'reason': result.fallback_reason.value if result.fallback_reason else None
            })
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        total = len(self.predictions)
        fallbacks = len(self.fallback_events)
        
        return {
            'total_predictions': total,
            'fallback_count': fallbacks,
            'fallback_rate': fallbacks / max(total, 1),
            'avg_latency_ms': np.mean([p['latency_ms'] for p in self.predictions]) if self.predictions else 0,
            'fallback_reasons': self._count_reasons()
        }
    
    def _count_reasons(self) -> Dict[str, int]:
        """Count fallback reasons"""
        reasons = {}
        for event in self.fallback_events:
            reason = event.get('reason', 'unknown')
            reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
