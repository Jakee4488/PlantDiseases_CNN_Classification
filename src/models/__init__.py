"""
Model factory for plant disease classification.
Provides easy access to different CNN architectures.
"""

import tensorflow as tf
from typing import Tuple

from .custom_cnn import create_custom_cnn
from .vggnet import create_vggnet
from .alexnet import create_alexnet
from .resnet import create_resnet


# Model registry mapping names to factory functions
MODEL_REGISTRY = {
    'custom_cnn': create_custom_cnn,
    'vggnet': create_vggnet,
    'alexnet': create_alexnet,
    'resnet': create_resnet,
}


def get_model(
    model_name: str,
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 3
) -> tf.keras.Model:
    """
    Get model by name from registry.
    
    Args:
        model_name: Name of the model architecture
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Keras model instance
        
    Raises:
        ValueError: If model_name is not in registry
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available_models}"
        )
    
    model_fn = MODEL_REGISTRY[model_name]
    return model_fn(input_shape=input_shape, num_classes=num_classes)


def list_available_models():
    """List all available model architectures."""
    return list(MODEL_REGISTRY.keys())
