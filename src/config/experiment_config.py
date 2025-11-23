"""
Experiment configuration management for reproducible training.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for model training experiments."""
    
    # Model configuration
    model_name: str = 'custom_cnn'
    input_shape: tuple = (256, 256, 3)
    num_classes: int = 3
    
    # Training configuration
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'categorical_crossentropy'
    metrics: list = None
    
    # Data configuration
    batch_size: int = 32
    epochs: int = 10
    validation_split: float = 0.0
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_seed: int = 123
    
    # Callbacks
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_monitor: str = 'val_accuracy'
    
    use_model_checkpoint: bool = True
    checkpoint_monitor: str = 'val_accuracy'
    checkpoint_mode: str = 'max'
    
    use_tensorboard: bool = True
    
    # Paths
    data_dir: str = 'Dataset'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    # MLflow
    experiment_name: str = 'plant_disease_classification'
    run_name: Optional[str] = None
    
    # Other
    seed: int = 123
    gpu_memory_growth: bool = True
    
    def __post_init__(self):
        """Post-initialization to set default values."""
        if self.metrics is None:
            self.metrics = ['accuracy']
        if self.run_name is None:
            self.run_name = f'{self.model_name}_experiment'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Convert tuples to lists for JSON serialization
        if 'input_shape' in config_dict:
            config_dict['input_shape'] = list(config_dict['input_shape'])
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        # Convert lists back to tuples where needed
        if 'input_shape' in config_dict:
            config_dict['input_shape'] = tuple(config_dict['input_shape'])
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_mlflow_params(self) -> Dict[str, Any]:
        """Get parameters to log to MLflow."""
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'use_augmentation': self.use_augmentation,
            'seed': self.seed,
        }
