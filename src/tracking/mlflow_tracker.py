"""
MLflow tracking integration for experiment management.
Handles logging of parameters, metrics, models, and artifacts.
"""

import mlflow
import mlflow.tensorflow
from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf


class MLflowTracker:
    """MLflow experiment tracker for model training."""
    
    def __init__(
        self,
        experiment_name: str = 'plant_disease_classification',
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
        """
        if tracking_uri is None:
            tracking_uri = 'file:./mlruns'
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        self.run = None
        
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Optional tags for the run
        """
        self.run = mlflow.start_run(run_name=run_name, tags=tags or {})
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (e.g., epoch)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: tf.keras.Model,
        artifact_path: str = 'model',
        registered_model_name: Optional[str] = None
    ):
        """
        Log TensorFlow model to MLflow.
        
        Args:
            model: Keras model to log
            artifact_path: Path within the run to store the model
            registered_model_name: Name to register the model under
        """
        mlflow.tensorflow.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to local file/directory
            artifact_path: Optional path within the run artifacts directory
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary: Dict, filename: str):
        """Log a dictionary as a JSON artifact."""
        mlflow.log_dict(dictionary, filename)
    
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure."""
        mlflow.log_figure(figure, artifact_file)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set tags for the current run."""
        for key, value in tags.items():
            mlflow.set_tag(key, value)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        self.run = None
    
    def get_run_id(self) -> Optional[str]:
        """Get the current run ID."""
        if self.run:
            return self.run.info.run_id
        return None
    
    @staticmethod
    def load_model(model_uri: str) -> tf.keras.Model:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: URI to the model (e.g., 'models:/model_name/production')
            
        Returns:
            Loaded Keras model
        """
        return mlflow.tensorflow.load_model(model_uri)


class MLflowCallback(tf.keras.callbacks.Callback):
    """Keras callback for logging to MLflow during training."""
    
    def __init__(self, tracker: MLflowTracker, log_every_n_epochs: int = 1):
        """
        Initialize MLflow callback.
        
        Args:
            tracker: MLflowTracker instance
            log_every_n_epochs: Log metrics every N epochs
        """
        super().__init__()
        self.tracker = tracker
        self.log_every_n_epochs = log_every_n_epochs
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics at the end of each epoch."""
        if logs and (epoch + 1) % self.log_every_n_epochs == 0:
            self.tracker.log_metrics(logs, step=epoch)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Log final metrics when training completes."""
        if logs:
            final_metrics = {f'final_{k}': v for k, v in logs.items()}
            self.tracker.log_metrics(final_metrics)
