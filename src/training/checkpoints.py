"""
Checkpoints Pattern Implementation
Enables training resumption from saved state in case of failure
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages training checkpoints for resumption after failures.

    Implements the Checkpoints Pattern from ML Design Patterns:
    - Saves model state, optimizer state, and training metrics
    - Enables resumption from last checkpoint
    - Automatic cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_frequency: int = 1,  # Save every N epochs
        experiment_name: str = "experiment",
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_frequency: Save checkpoint every N epochs
            experiment_name: Name of the experiment for checkpoint naming
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_frequency = save_frequency
        self.experiment_name = experiment_name

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoints
        self.checkpoints: List[str] = []
        self._load_checkpoint_history()

    def _load_checkpoint_history(self):
        """Load existing checkpoint history"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                self.checkpoints = json.load(f)

    def _save_checkpoint_history(self):
        """Save checkpoint history"""
        history_file = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_file, "w") as f:
            json.dump(self.checkpoints, f, indent=2)

    def save_checkpoint(
        self,
        epoch: int,
        model: Any,
        optimizer_state: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        training_state: Optional[Dict] = None,
        extra_data: Optional[Dict] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number
            model: Model object to save
            optimizer_state: Optimizer state dictionary
            metrics: Training metrics dictionary
            training_state: Additional training state (e.g., best score)
            extra_data: Any extra data to save

        Returns:
            Path to saved checkpoint
        """
        # Check if we should save at this epoch
        if epoch % self.save_frequency != 0:
            logger.debug(
                f"Skipping checkpoint at epoch {epoch} (frequency={self.save_frequency})"
            )
            return None

        # Create checkpoint filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.experiment_name}_epoch_{epoch}_{timestamp}.ckpt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Create checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "timestamp": timestamp,
            "experiment_name": self.experiment_name,
            "model_state": model,
            "optimizer_state": optimizer_state,
            "metrics": metrics or {},
            "training_state": training_state or {},
            "extra_data": extra_data or {},
        }

        # Save checkpoint
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update checkpoint history
        self.checkpoints.append(str(checkpoint_path))
        self._save_checkpoint_history()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to specific checkpoint. If None, loads latest.

        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning("No checkpoint found to load")
            return None

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint_data['epoch']}")

        return checkpoint_data

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint"""
        if not self.checkpoints:
            return None

        # Find existing checkpoints
        existing = [cp for cp in self.checkpoints if os.path.exists(cp)]

        if not existing:
            return None

        return existing[-1]

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit"""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

        self._save_checkpoint_history()

    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists"""
        return self.get_latest_checkpoint() is not None


class TrainingResumer:
    """
    High-level interface for training with checkpoint support.

    Usage:
        resumer = TrainingResumer(checkpoint_dir="checkpoints")

        # Check if resuming
        if resumer.can_resume():
            state = resumer.resume()
            start_epoch = state['epoch'] + 1
            model = state['model_state']
        else:
            start_epoch = 0
            model = create_new_model()

        # Training loop
        for epoch in range(start_epoch, total_epochs):
            train_one_epoch(model)
            resumer.save(epoch, model, metrics)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        experiment_name: str = "training",
        max_checkpoints: int = 3,
        save_frequency: int = 1,
    ):
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=max_checkpoints,
            save_frequency=save_frequency,
            experiment_name=experiment_name,
        )

        self.current_best_score = None
        self.current_epoch = 0

    def can_resume(self) -> bool:
        """Check if training can be resumed"""
        return self.checkpoint_manager.has_checkpoint()

    def resume(self) -> Dict:
        """Resume training from last checkpoint"""
        checkpoint = self.checkpoint_manager.load_checkpoint()

        if checkpoint:
            self.current_epoch = checkpoint["epoch"]
            training_state = checkpoint.get("training_state", {})
            self.current_best_score = training_state.get("best_score")

        return checkpoint

    def save(self, epoch: int, model: Any, metrics: Dict, is_best: bool = False) -> str:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Training metrics
            is_best: Whether this is the best model so far
        """
        training_state = {"best_score": self.current_best_score, "is_best": is_best}

        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            epoch=epoch, model=model, metrics=metrics, training_state=training_state
        )

        self.current_epoch = epoch

        return checkpoint_path


class ModelCheckpointCallback:
    """
    Callback for saving model checkpoints during training.
    Works with scikit-learn style training loops.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        monitor: str = "val_f1",
        mode: str = "max",
        save_best_only: bool = True,
        experiment_name: str = "model",
    ):
        """
        Initialize callback.

        Args:
            checkpoint_dir: Directory for checkpoints
            monitor: Metric to monitor
            mode: 'max' or 'min' - whether higher or lower is better
            save_best_only: Only save when monitored metric improves
            experiment_name: Name for checkpoint files
        """
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir, experiment_name=experiment_name
        )
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only

        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.best_checkpoint = None

    def on_epoch_end(self, epoch: int, model: Any, metrics: Dict):
        """Called at the end of each epoch"""
        current_score = metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"Monitored metric '{self.monitor}' not found in metrics")
            return

        # Check if this is the best score
        is_best = False
        if self.mode == "max" and current_score > self.best_score:
            is_best = True
        elif self.mode == "min" and current_score < self.best_score:
            is_best = True

        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                metrics=metrics,
                training_state={"best_score": self.best_score, "is_best": is_best},
            )

            if is_best:
                self.best_score = current_score
                self.best_checkpoint = checkpoint_path
                logger.info(f"New best {self.monitor}: {current_score:.4f}")

    def get_best_model(self) -> Optional[Any]:
        """Load and return the best model"""
        if self.best_checkpoint:
            checkpoint = self.checkpoint_manager.load_checkpoint(self.best_checkpoint)
            return checkpoint.get("model_state")
        return None
