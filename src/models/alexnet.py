"""
AlexNet architecture implementation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def create_alexnet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 3
) -> tf.keras.Model:
    """
    Create AlexNet-style architecture.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', input_shape=input_shape),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        
        # Layer 2
        layers.Conv2D(256, (5, 5), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        layers.BatchNormalization(),
        
        # Layer 3
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.Activation('relu'),
        
        # Layer 4
        layers.Conv2D(384, (3, 3), padding='same'),
        layers.Activation('relu'),
        
        # Layer 5
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((3, 3), strides=(2, 2)),
        
        # Classification
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='alexnet')
    
    return model
