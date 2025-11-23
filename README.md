# MLOps Plant Disease Classification System

A production-ready MLOps system for classifying plant diseases (Healthy, Powdery, Rust) using Convolutional Neural Networks (CNNs). This project transforms a research notebook into a scalable, reproducible, and automated machine learning pipeline.

## 🏗 Architecture

The system follows MLOps best practices with the following components:

1.  **Data Management**: DVC for dataset versioning and lineage.
2.  **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts.
3.  **Model Development**: Modular TensorFlow/Keras implementation of VGGNet, AlexNet, ResNet, and Custom CNN.
4.  **CI/CT Pipeline**: GitHub Actions for automated testing and continuous training.
5.  **Deployment**: FastAPI application containerized with Docker and deployed on Kubernetes.
6.  **Monitoring**: Prometheus metrics and Evidently AI for data drift detection.
7.  **Orchestration**: Apache Airflow for automated retraining pipelines.

## 📂 Project Structure

```
PlantDiseases_CNN_Classification/
├── src/                    # Source code
│   ├── data/              # Data loading & augmentation (DVC integrated)
│   ├── models/            # Model architectures (VGG, AlexNet, ResNet)
│   ├── config/            # Configuration management
│   ├── tracking/          # MLflow tracking utilities
│   ├── utils/             # Visualization & helpers
│   ├── train.py           # Main training script
│   └── tune.py            # Hyperparameter tuning script
├── serve/                 # Model serving
│   ├── app.py             # FastAPI application
│   └── Dockerfile         # Container definition
├── deploy/                # Deployment configurations
│   └── kubernetes/        # K8s manifests (Deployment, Service, Ingress)
├── monitoring/            # Observability
│   └── data_drift_detector.py # Evidently AI drift detection
├── pipelines/             # Orchestration
│   └── retraining_pipeline.py # Airflow DAGs
├── tests/                 # Unit & integration tests
├── configs/               # YAML experiment configs
├── .github/workflows/     # CI/CT pipelines
└── requirements.txt       # Project dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Kubernetes (optional for local dev)
- Git

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd PlantDiseases_CNN_Classification
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize DVC** (if starting fresh):
    ```bash
    dvc init
    # Configure remote storage (e.g., Azure Blob / S3)
    # dvc remote add -d storage s3://my-bucket/data
    ```

### Training a Model

Run the training pipeline with default configuration:

```bash
python src/train.py
```

To use a specific model architecture:

```bash
python src/train.py --model vggnet --epochs 20
```

### Running the API

Start the FastAPI server locally:

```bash
uvicorn serve.app:app --reload
```

Access the API documentation at `http://localhost:8000/docs`.

## 📊 MLOps Workflows

### Experiment Tracking
Launch the MLflow UI to view experiments:
```bash
mlflow ui
```

### Hyperparameter Tuning
Run automated grid search:
```bash
python src/tune.py --epochs 5
```

### Deployment
Build and deploy to Kubernetes:
```bash
docker build -t plant-disease-classifier:latest -f serve/Dockerfile .
kubectl apply -f deploy/kubernetes/
```

## 📈 Monitoring

- **Metrics**: Prometheus metrics available at `/metrics`
- **Drift Detection**: Run `python monitoring/data_drift_detector.py` to generate reports

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a Pull Request

## 📄 License
[MIT License](LICENSE)
