"""
Model factory for plant disease classification.
Provides easy access to different CNN architectures.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def create_custom_cnn(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 3
) -> tf.keras.Model:
    """
    Create custom CNN architecture for plant disease classification.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), strides=1, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(256, (5, 5), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fifth conv block
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='custom_cnn')
    
    return model


# Model registry mapping names to factory functions
MODEL_REGISTRY = {
    'custom_cnn': create_custom_cnn,
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
