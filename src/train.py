"""
Main training script with MLOps integration.
Trains plant disease classification model with MLflow tracking.
"""

import tensorflow as tf
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data import PlantDiseaseDataLoader
from models import get_model
from config import ExperimentConfig
from tracking import MLflowTracker, MLflowCallback


def setup_gpu(memory_growth: bool = True):
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'GPU found: {len(gpus)} device(s)')
        except RuntimeError as e:
            print(f'GPU setup error: {e}')
    else:
        print('No GPU found, using CPU')


def train_model(config: ExperimentConfig):
    """
    Train model with given configuration.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Trained model and history
    """
    # Setup GPU
    setup_gpu(config.gpu_memory_growth)
    
    # Initialize MLflow tracker
    tracker = MLflowTracker(experiment_name=config.experiment_name)
    
    # Start MLflow run
    with tracker.start_run(run_name=config.run_name):
        print(f'\n=== Starting experiment: {config.run_name} ===\n')
        
        # Log parameters
        tracker.log_params(config.get_mlflow_params())
        tracker.log_dict(config.to_dict(), 'config.json')
        
        # Load data
        print('Loading datasets...')
        data_loader = PlantDiseaseDataLoader(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            seed=config.seed
        )
        
        train_ds = data_loader.load_dataset(
            split='train',
            shuffle=True,
            augment=config.use_augmentation
        )
        val_ds = data_loader.load_dataset(
            split='validation',
            shuffle=False,
            augment=False
        )
        
        # Log dataset info
        train_info = data_loader.get_dataset_info('train')
        val_info = data_loader.get_dataset_info('validation')
        tracker.log_dict(train_info, 'train_dataset_info.json')
        tracker.log_dict(val_info, 'val_dataset_info.json')
        
        print(f'Train samples: {train_info["total_images"]}')
        print(f'Validation samples: {val_info["total_images"]}')
        
        # Build model
        print(f'\nBuilding model: {config.model_name}')
        model = get_model(
            model_name=config.model_name,
            input_shape=config.input_shape,
            num_classes=config.num_classes
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.get({
                'class_name': config.optimizer,
                'config': {'learning_rate': config.learning_rate}
            }),
            loss=config.loss,
            metrics=config.metrics
        )
        
        print(f'\nModel summary:')
        model.summary()
        
        # Setup callbacks
        callbacks = []
        
        # MLflow callback
        callbacks.append(MLflowCallback(tracker))
        
        # Early stopping
        if config.use_early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor=config.early_stopping_monitor,
                patience=config.early_stopping_patience,
                mode='max' if 'acc' in config.early_stopping_monitor else 'min',
                restore_best_weights=True,
                verbose=1
            ))
        
        # Model checkpoint
        if config.use_model_checkpoint:
            checkpoint_path = Path(config.checkpoint_dir) / f'{config.model_name}_best.h5'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor=config.checkpoint_monitor,
                mode=config.checkpoint_mode,
                save_best_only=True,
                verbose=1
            ))
        
        # TensorBoard
        if config.use_tensorboard:
            tensorboard_dir = Path(config.log_dir) / config.run_name
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(tf.keras.callbacks.TensorBoard(
                log_dir=str(tensorboard_dir),
                histogram_freq=1
            ))
        
        # Train model
        print(f'\nStarting training for {config.epochs} epochs...\n')
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print('\nEvaluating on test set...')
        test_ds = data_loader.load_dataset(split='test', shuffle=False, augment=False)
        test_results = model.evaluate(test_ds, verbose=1)
        
        # Log test metrics
        test_metrics = {
            f'test_{metric}': value
            for metric, value in zip(model.metrics_names, test_results)
        }
        tracker.log_metrics(test_metrics)
        
        # Generate predictions for visualization
        print('\nGenerating visualizations...')
        y_pred_proba = model.predict(test_ds)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = np.concatenate([y for x, y in test_ds], axis=0)
        y_true_indices = np.argmax(y_true, axis=1)
        
        # Plot and log confusion matrix
        from utils import plot_confusion_matrix, plot_roc_curves, plot_training_history
        
        cm_fig = plot_confusion_matrix(
            y_true_indices,
            y_pred,
            data_loader.class_names
        )
        tracker.log_figure(cm_fig, 'confusion_matrix.png')
        plt.close(cm_fig)
        
        # Plot and log ROC curves
        roc_fig = plot_roc_curves(
            y_true,
            y_pred_proba,
            data_loader.class_names
        )
        tracker.log_figure(roc_fig, 'roc_curves.png')
        plt.close(roc_fig)
        
        # Plot and log training history
        history_fig = plot_training_history(history.history)
        tracker.log_figure(history_fig, 'training_history.png')
        plt.close(history_fig)
        
        # Save model to MLflow
        print('\nSaving model to MLflow...')
        tracker.log_model(
            model,
            artifact_path='model',
            registered_model_name=f'{config.model_name}_production'
        )
        
        print(f'\n=== Training complete ===')
        print(f'Run ID: {tracker.get_run_id()}')
        print(f'Test accuracy: {test_metrics.get("test_accuracy", "N/A"):.4f}')
        
        return model, history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train plant disease classification model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model name (overrides config)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Override with command line arguments
    if args.model:
        config.model_name = args.model
        config.run_name = f'{args.model}_experiment'
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
