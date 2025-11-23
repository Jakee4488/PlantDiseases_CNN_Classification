# MLOps Plant Disease Classification

Production-ready MLOps pipeline for CNN-based plant disease classification with comprehensive experiment tracking, automated deployment, and continuous monitoring.

## Project Structure

```
PlantDiseases_CNN_Classification/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── config/            # Configuration management
│   ├── tracking/          # MLflow tracking utilities
│   └── utils/             # Helper functions
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── serve/                 # API serving
├── deploy/                # Deployment configs
├── monitoring/            # Monitoring and drift detection
├── pipelines/             # Training pipelines
├── configs/               # Experiment configs
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── mlruns/               # MLflow experiments

```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize DVC
```bash
dvc init
dvc remote add -d storage <your-cloud-storage>
```

### 3. Train Model
```bash
python src/train.py --config configs/experiment.yaml
```

### 4. Start MLflow UI
```bash
mlflow ui
```

### 5. Serve Model
```bash
uvicorn serve.app:app --reload
```

## MLOps Components

- **Data Versioning**: DVC for dataset tracking
- **Experiment Tracking**: MLflow for metrics and model registry
- **CI/CT**: GitHub Actions for automated testing and training
- **Deployment**: FastAPI + Kubernetes
- **Monitoring**: Prometheus + Grafana with drift detection
- **Retraining**: Apache Airflow for automated pipelines

## Dataset

Plant Diseases Dataset with 3 classes:
- Healthy
- Powdery
- Rust

**Source**: [Kaggle Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

## License

See LICENSE file for details.
