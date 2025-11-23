"""
Unit tests for PlantDiseaseDataLoader.
"""

import pytest
import tensorflow as tf
from pathlib import Path
import sys

from src.data import PlantDiseaseDataLoader


@pytest.fixture
def data_loader():
    """Create data loader fixture."""
    return PlantDiseaseDataLoader(
        data_dir='Dataset',
        img_size=(256, 256),
        batch_size=32,
        seed=123
    )


def test_data_loader_initialization(data_loader):
    """Test data loader initialization."""
    assert data_loader.img_size == (256, 256)
    assert data_loader.batch_size == 32
    assert data_loader.seed == 123
    assert len(data_loader.class_names) == 3


def test_augmentation_pipeline():
    """Test data augmentation pipeline."""
    image = tf.random.uniform((256, 256, 3))
    loader = PlantDiseaseDataLoader()
    augmented = loader._augment(image)
    
    assert augmented.shape == (256, 256, 3)
    assert tf.reduce_min(augmented) >= 0.0
    assert tf.reduce_max(augmented) <= 1.0


def test_get_dataset_info(data_loader):
    """Test dataset info retrieval."""
    # This test will skip if dataset doesn't exist
    try:
        info = data_loader.get_dataset_info('train')
        assert 'split' in info
        assert 'classes' in info
        assert 'num_classes' in info
        assert info['num_classes'] == 3
    except FileNotFoundError:
        pytest.skip("Dataset not found")


def test_class_names(data_loader):
    """Test class names are correct."""
    expected_classes = ['Healthy', 'Powdery', 'Rust']
    assert data_loader.class_names == expected_classes
