"""
Unit tests for model factory.
"""

import pytest
import tensorflow as tf
from pathlib import Path
import sys

from src.models import get_model, list_available_models


def test_get_custom_cnn():
    """Test creating custom CNN model."""
    model = get_model('custom_cnn', input_shape=(256, 256, 3), num_classes=3)
    
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 256, 256, 3)
    assert model.output_shape == (None, 3)


def test_invalid_model_name():
    """Test error handling for invalid model name."""
    with pytest.raises(ValueError, match='Unknown model'):
        get_model('nonexistent_model')


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert 'custom_cnn' in models
    assert 'vggnet' in models
    assert 'alexnet' in models
    assert 'resnet' in models


def test_model_compilation():
    """Test model can be compiled."""
    for model_name in ['custom_cnn', 'vggnet', 'resnet']:
        model = get_model(model_name, input_shape=(256, 256, 3), num_classes=3)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        assert model.optimizer is not None
        assert model.loss is not None
