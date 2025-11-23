"""
VGGNet architecture implementation.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple


def create_vggnet(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    num_classes: int = 3
) -> tf.keras.Model:
    """
    Create VGGNet-style architecture.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 2
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 3
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 4
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Block 5
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        
        # Classification block
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='vggnet')
    
    return model
