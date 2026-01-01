"""
Production Monitoring Utilities
Monitoring and alerting for ML model performance in production.
"""

import json
import logging
import os
import warnings
from collections import deque
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelMonitor")


class PredictionLogger:
    """
    Logs predictions and ground truth for monitoring.
    """

    def __init__(self, log_dir: str = "prediction_logs", max_buffer: int = 10000):
        self.log_dir = log_dir
        self.max_buffer = max_buffer
        self.predictions_buffer = deque(maxlen=max_buffer)
        os.makedirs(log_dir, exist_ok=True)

    def log_prediction(
        self,
        prediction: int,
        probability: float,
        features: Optional[np.ndarray] = None,
        ground_truth: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log a single prediction.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": int(prediction),
            "probability": float(probability),
            "ground_truth": int(ground_truth) if ground_truth is not None else None,
            "metadata": metadata or {},
        }

        if features is not None:
            entry["features_hash"] = hash(features.tobytes())

        self.predictions_buffer.append(entry)

    def log_batch(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ):
        """
        Log a batch of predictions.
        """
        for i in range(len(predictions)):
            gt = ground_truth[i] if ground_truth is not None else None
            self.log_prediction(predictions[i], probabilities[i], ground_truth=gt)

    def flush_to_disk(self, filename: Optional[str] = None):
        """
        Write buffer to disk.
        """
        if not self.predictions_buffer:
            return

        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        path = os.path.join(self.log_dir, filename)

        with open(path, "a") as f:
            for entry in self.predictions_buffer:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Flushed {len(self.predictions_buffer)} predictions to {path}")
        self.predictions_buffer.clear()

    def get_recent_predictions(self, n: int = 100) -> List[dict]:
        """
        Get recent predictions from buffer.
        """
        return list(self.predictions_buffer)[-n:]


class DriftDetector:
    """
    Detects data and concept drift in production.
    """

    def __init__(self, reference_data: np.ndarray = None, window_size: int = 1000):
        self.reference_data = reference_data
        self.reference_stats = None
        self.window_size = window_size
        self.current_window = deque(maxlen=window_size)
        self.drift_history = []

        if reference_data is not None:
            self._compute_reference_stats(reference_data)

    def _compute_reference_stats(self, data: np.ndarray):
        """
        Compute reference statistics for drift detection.
        """
        self.reference_stats = {
            "mean": np.mean(data, axis=0),
            "std": np.std(data, axis=0),
            "min": np.min(data, axis=0),
            "max": np.max(data, axis=0),
            "percentiles": {
                "p25": np.percentile(data, 25, axis=0),
                "p50": np.percentile(data, 50, axis=0),
                "p75": np.percentile(data, 75, axis=0),
            },
        }

    def set_reference(self, data: np.ndarray):
        """
        Set reference data for drift detection.
        """
        self.reference_data = data
        self._compute_reference_stats(data)
        logger.info(f"Reference data set with {len(data)} samples")

    def add_sample(self, sample: np.ndarray):
        """
        Add a sample to the current window.
        """
        self.current_window.append(sample)

    def detect_drift(self, threshold: float = 0.1) -> dict:
        """
        Detect drift between reference and current window.
        Uses Population Stability Index (PSI) and distribution comparison.
        """
        if self.reference_stats is None:
            raise ValueError("Reference data not set")

        if len(self.current_window) < 100:
            return {"drift_detected": False, "reason": "Insufficient samples"}

        current_data = np.array(list(self.current_window))

        # Compute current statistics
        current_stats = {
            "mean": np.mean(current_data, axis=0),
            "std": np.std(current_data, axis=0),
        }

        # Mean shift detection
        mean_shift = np.abs(current_stats["mean"] - self.reference_stats["mean"]) / (
            self.reference_stats["std"] + 1e-10
        )

        # Variance change detection
        std_ratio = current_stats["std"] / (self.reference_stats["std"] + 1e-10)

        # Aggregate drift score
        drift_score = np.mean(mean_shift) + np.abs(np.mean(std_ratio) - 1)

        drift_detected = drift_score > threshold

        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "mean_shift_avg": float(np.mean(mean_shift)),
            "std_ratio_avg": float(np.mean(std_ratio)),
            "features_drifted": int(np.sum(mean_shift > threshold)),
            "timestamp": datetime.now().isoformat(),
        }

        if drift_detected:
            logger.warning(f"Data drift detected! Score: {drift_score:.4f}")

        self.drift_history.append(result)
        return result

    def get_drift_history(self) -> List[dict]:
        """
        Get drift detection history.
        """
        return self.drift_history


class PerformanceMonitor:
    """
    Monitors model performance metrics over time.
    """

    def __init__(self, baseline_metrics: dict = None, alert_threshold: float = 0.05):
        self.baseline_metrics = baseline_metrics or {}
        self.alert_threshold = alert_threshold
        self.metrics_history = []
        self.alerts = []

    def set_baseline(self, metrics: dict):
        """
        Set baseline performance metrics.
        """
        self.baseline_metrics = metrics
        logger.info(f"Baseline metrics set: {metrics}")

    def log_metrics(self, metrics: dict, window_name: str = None):
        """
        Log performance metrics.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "window": window_name,
            "metrics": metrics,
        }

        self.metrics_history.append(entry)

        # Check for degradation
        self._check_degradation(metrics)

    def _check_degradation(self, current_metrics: dict):
        """
        Check for performance degradation.
        """
        if not self.baseline_metrics:
            return

        for metric, value in current_metrics.items():
            if metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                degradation = (baseline - value) / baseline

                if degradation > self.alert_threshold:
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "metric": metric,
                        "baseline": baseline,
                        "current": value,
                        "degradation_pct": degradation * 100,
                        "severity": "HIGH" if degradation > 0.1 else "MEDIUM",
                    }
                    self.alerts.append(alert)
                    logger.warning(
                        f"Performance degradation alert: {metric} "
                        f"dropped from {baseline:.4f} to {value:.4f} "
                        f"({degradation*100:.2f}% degradation)"
                    )

    def get_alerts(self, severity: str = None) -> List[dict]:
        """
        Get alerts, optionally filtered by severity.
        """
        if severity:
            return [a for a in self.alerts if a["severity"] == severity]
        return self.alerts

    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get summary of metrics history.
        """
        if not self.metrics_history:
            return pd.DataFrame()

        records = []
        for entry in self.metrics_history:
            record = {"timestamp": entry["timestamp"], "window": entry["window"]}
            record.update(entry["metrics"])
            records.append(record)

        return pd.DataFrame(records)


class ModelMonitor:
    """
    Comprehensive model monitoring combining all monitoring capabilities.
    """

    def __init__(self, model_name: str, output_dir: str = "monitoring"):
        self.model_name = model_name
        self.output_dir = output_dir

        self.prediction_logger = PredictionLogger(
            os.path.join(output_dir, "predictions")
        )
        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Model monitor initialized for: {model_name}")

    def set_reference_data(
        self, X_reference: np.ndarray, baseline_metrics: dict = None
    ):
        """
        Set reference data and baseline metrics.
        """
        self.drift_detector.set_reference(X_reference)

        if baseline_metrics:
            self.performance_monitor.set_baseline(baseline_metrics)

    def log_inference(
        self,
        features: np.ndarray,
        prediction: int,
        probability: float,
        ground_truth: int = None,
    ):
        """
        Log a single inference.
        """
        self.prediction_logger.log_prediction(
            prediction, probability, features, ground_truth
        )
        self.drift_detector.add_sample(features)

    def check_health(self) -> dict:
        """
        Perform health check.
        """
        health = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "status": "HEALTHY",
            "checks": {},
        }

        # Check drift
        try:
            drift_result = self.drift_detector.detect_drift()
            health["checks"]["drift"] = drift_result
            if drift_result.get("drift_detected"):
                health["status"] = "WARNING"
        except:
            health["checks"]["drift"] = {"status": "SKIPPED"}

        # Check alerts
        high_alerts = self.performance_monitor.get_alerts(severity="HIGH")
        health["checks"]["alerts"] = {
            "high_severity_count": len(high_alerts),
            "total_alerts": len(self.performance_monitor.alerts),
        }

        if len(high_alerts) > 0:
            health["status"] = "CRITICAL"

        return health

    def generate_report(self) -> dict:
        """
        Generate comprehensive monitoring report.
        """
        report = {
            "model_name": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "predictions_logged": len(self.prediction_logger.predictions_buffer),
            "drift_checks": len(self.drift_detector.drift_history),
            "alerts": self.performance_monitor.get_alerts(),
            "health": self.check_health(),
        }

        # Save report
        report_path = os.path.join(
            self.output_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {report_path}")
        return report

    def flush_logs(self):
        """
        Flush all logs to disk.
        """
        self.prediction_logger.flush_to_disk()


def create_monitor_for_production(
    model_name: str, X_reference: np.ndarray, baseline_metrics: dict
) -> ModelMonitor:
    """
    Factory function to create a configured monitor for production.
    """
    monitor = ModelMonitor(model_name)
    monitor.set_reference_data(X_reference, baseline_metrics)
    return monitor
