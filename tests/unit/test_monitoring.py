"""
Unit Tests for Monitoring Module
Tests for Drift Detection, Performance Monitoring, CME patterns
"""
import pytest
import numpy as np
from datetime import datetime


class TestDriftDetector:
    """Tests for DriftDetector (Continued Model Evaluation Pattern)"""
    
    @pytest.fixture
    def reference_data(self):
        """Create reference data"""
        np.random.seed(42)
        return np.random.normal(loc=0, scale=1, size=(1000, 10))
    
    def test_set_reference(self, reference_data):
        """Test setting reference data"""
        from src.monitoring.production_monitor import DriftDetector
        
        detector = DriftDetector()
        detector.set_reference(reference_data)
        
        assert detector.reference_data is not None
        assert detector.reference_stats is not None
        assert 'mean' in detector.reference_stats
        assert 'std' in detector.reference_stats
    
    def test_no_drift_similar_distribution(self, reference_data):
        """Test no drift detected for similar distribution"""
        from src.monitoring.production_monitor import DriftDetector
        
        detector = DriftDetector(reference_data=reference_data)
        
        # Add samples from same distribution
        np.random.seed(123)
        for _ in range(500):
            sample = np.random.normal(loc=0, scale=1, size=10)
            detector.add_sample(sample)
        
        result = detector.detect_drift(threshold=0.5)
        
        assert result['drift_detected'] == False
    
    def test_drift_detected_different_distribution(self, reference_data):
        """Test drift detected for different distribution"""
        from src.monitoring.production_monitor import DriftDetector
        
        detector = DriftDetector(reference_data=reference_data)
        
        # Add samples from different distribution (shifted mean)
        np.random.seed(123)
        for _ in range(500):
            sample = np.random.normal(loc=5, scale=2, size=10)  # Shifted
            detector.add_sample(sample)
        
        result = detector.detect_drift(threshold=0.5)
        
        assert result['drift_detected'] == True
        assert result['drift_score'] > 0.5
    
    def test_insufficient_samples(self, reference_data):
        """Test handling of insufficient samples"""
        from src.monitoring.production_monitor import DriftDetector
        
        detector = DriftDetector(reference_data=reference_data)
        
        # Add only few samples
        for _ in range(10):
            detector.add_sample(np.random.normal(size=10))
        
        result = detector.detect_drift()
        
        assert result['drift_detected'] == False
        assert 'Insufficient' in result.get('reason', '')


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor"""
    
    def test_set_baseline(self):
        """Test setting baseline metrics"""
        from src.monitoring.production_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        baseline = {'accuracy': 0.85, 'f1_score': 0.78, 'roc_auc': 0.90}
        
        monitor.set_baseline(baseline)
        
        assert monitor.baseline_metrics == baseline
    
    def test_log_metrics(self):
        """Test logging metrics"""
        from src.monitoring.production_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        metrics = {'accuracy': 0.82, 'f1_score': 0.75}
        
        monitor.log_metrics(metrics, window_name='test_window')
        
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0]['metrics'] == metrics
    
    def test_degradation_alert(self):
        """Test degradation alert is triggered"""
        from src.monitoring.production_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(
            baseline_metrics={'accuracy': 0.90},
            alert_threshold=0.05
        )
        
        # Log metrics with significant degradation
        monitor.log_metrics({'accuracy': 0.80})  # 11% drop
        
        assert len(monitor.alerts) > 0
        assert monitor.alerts[0]['metric'] == 'accuracy'
        assert monitor.alerts[0]['severity'] == 'HIGH'
    
    def test_no_alert_within_threshold(self):
        """Test no alert when within threshold"""
        from src.monitoring.production_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(
            baseline_metrics={'accuracy': 0.90},
            alert_threshold=0.05
        )
        
        # Log metrics with small degradation
        monitor.log_metrics({'accuracy': 0.88})  # 2% drop
        
        assert len(monitor.alerts) == 0


class TestPredictionLogger:
    """Tests for PredictionLogger"""
    
    def test_log_prediction(self):
        """Test logging a prediction"""
        from src.monitoring.production_monitor import PredictionLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir)
            
            logger.log_prediction(
                prediction=1,
                probability=0.85,
                ground_truth=1
            )
            
            assert len(logger.predictions_buffer) == 1
    
    def test_log_batch(self):
        """Test logging batch predictions"""
        from src.monitoring.production_monitor import PredictionLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir)
            
            predictions = np.array([0, 1, 1, 0])
            probabilities = np.array([0.2, 0.8, 0.9, 0.3])
            
            logger.log_batch(predictions, probabilities)
            
            assert len(logger.predictions_buffer) == 4
    
    def test_buffer_max_size(self):
        """Test buffer respects max size"""
        from src.monitoring.production_monitor import PredictionLogger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir, max_buffer=10)
            
            for i in range(20):
                logger.log_prediction(prediction=1, probability=0.5)
            
            assert len(logger.predictions_buffer) == 10
