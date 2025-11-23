"""
Hyperparameter tuning script.
Runs multiple experiments with different configurations.
"""

import itertools
import argparse
from typing import Dict, List, Any
import copy
from src.config import ExperimentConfig
from src.train import train_model


def get_hyperparameter_grid() -> Dict[str, List[Any]]:
    """Define hyperparameter grid to search."""
    return {
        'model_name': ['custom_cnn', 'vggnet', 'resnet'],
        'learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],
        'optimizer': ['adam', 'sgd']
    }


def run_tuning(base_config_path: str, epochs: int = 5):
    """
    Run hyperparameter tuning.
    
    Args:
        base_config_path: Path to base configuration file
        epochs: Number of epochs per experiment
    """
    # Load base config
    base_config = ExperimentConfig.from_yaml(base_config_path)
    base_config.epochs = epochs
    
    # Get grid
    grid = get_hyperparameter_grid()
    keys = list(grid.keys())
    values = list(grid.values())
    
    # Generate combinations
    combinations = list(itertools.product(*values))
    print(f"Found {len(combinations)} combinations to test.")
    
    for i, combo in enumerate(combinations):
        print(f"\n=== Running experiment {i+1}/{len(combinations)} ===")
        
        # Create config for this run
        config = copy.deepcopy(base_config)
        params = dict(zip(keys, combo))
        
        # Update config
        for key, value in params.items():
            setattr(config, key, value)
            
        # Set run name
        config.run_name = f"tuning_{i+1}_{'_'.join(f'{k}_{v}' for k, v in params.items())}"
        
        print(f"Configuration: {params}")
        
        try:
            train_model(config)
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to base configuration YAML file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of epochs per experiment'
    )
    
    args = parser.parse_args()
    run_tuning(args.config, args.epochs)


if __name__ == '__main__':
    main()
