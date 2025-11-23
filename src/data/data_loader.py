"""
Data loader module for Plant Disease Classification with DVC integration.
Handles loading, preprocessing, and augmentation of image data.
"""

import tensorflow as tf
from pathlib import Path
from typing import Tuple, Optional
import os


class PlantDiseaseDataLoader:
    """Data loader with versioning support for plant disease dataset."""
    
    def __init__(
        self,
        data_dir: str = 'Dataset',
        img_size: Tuple[int, int] = (256, 256),
        batch_size: int = 32,
        seed: int = 123
    ):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing train/test/validation folders
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.class_names = ['Healthy', 'Powdery', 'Rust']
        
    def load_dataset(
        self,
        split: str = 'train',
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """
        Load dataset for specified split.
        
        Args:
            split: One of 'train', 'test', 'validation'
            shuffle: Whether to shuffle the dataset
            augment: Whether to apply data augmentation
            
        Returns:
            tf.data.Dataset object
        """
        split_dir = self.data_dir / split.capitalize()
        
        if not split_dir.exists():
            raise ValueError(f"Directory not found: {split_dir}")
        
        # Create dataset from directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(split_dir),
            labels='inferred',
            label_mode='categorical',
            class_names=self.class_names,
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=shuffle,
            seed=self.seed
        )
        
        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
        
        # Apply augmentation if requested
        if augment and split == 'train':
            dataset = dataset.map(
                lambda x, y: (self._augment(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Performance optimization
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment(self, image: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation to image."""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Random rotation
        image = tf.image.rot90(
            image,
            k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        )
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Clip values to [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image
    
    def get_dataset_info(self, split: str = 'train') -> dict:
        """Get information about the dataset."""
        split_dir = self.data_dir / split.capitalize()
        
        info = {
            'split': split,
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'image_size': self.img_size,
            'batch_size': self.batch_size
        }
        
        # Count images per class
        class_counts = {}
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.jpg'))) + \
                       len(list(class_dir.glob('*.png')))
                class_counts[class_name] = count
        
        info['class_distribution'] = class_counts
        info['total_images'] = sum(class_counts.values())
        
        return info


def create_augmentation_pipeline(seed: int = 123) -> tf.keras.Sequential:
    """
    Create a Keras Sequential model for data augmentation.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        tf.keras.Sequential model with augmentation layers
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(0.2, seed=seed),
        tf.keras.layers.RandomZoom(0.1, seed=seed),
        tf.keras.layers.RandomContrast(0.1, seed=seed),
    ], name='augmentation')
