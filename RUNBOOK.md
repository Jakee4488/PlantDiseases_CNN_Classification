# MLOps Operational Runbook

This runbook provides detailed procedures for maintaining, operating, and troubleshooting the Plant Disease Classification MLOps system.

## 📋 Table of Contents
1. [Data Management Operations](#data-management-operations)
2. [Model Training & Experimentation](#model-training--experimentation)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Incident Response](#monitoring--incident-response)
5. [Maintenance Tasks](#maintenance-tasks)

---

## 1. Data Management Operations

### Adding New Data
**Trigger**: When new labeled images are available.

1.  **Organize Data**: Place new images in `data/raw/Dataset/{split}/{class}/`.
2.  **Update DVC**:
    ```bash
    dvc add data/raw/Dataset
    dvc commit
    dvc push
    ```
3.  **Commit Changes**:
    ```bash
    git add data/raw/Dataset.dvc
    git commit -m "Update dataset with new samples"
    git push
    ```

### Reverting to Previous Dataset
**Trigger**: Data corruption or poor model performance due to bad data.

1.  **Checkout DVC Version**:
    ```bash
    git checkout <commit-hash> data/raw/Dataset.dvc
    dvc checkout
    ```

---

## 2. Model Training & Experimentation

### Running a New Experiment
**Trigger**: Testing a new hypothesis or architecture.

1.  **Create Config**: Copy `configs/default_config.yaml` to `configs/experiment_name.yaml`.
2.  **Modify Parameters**: Update learning rate, batch size, or model architecture.
3.  **Run Training**:
    ```bash
    python src/train.py --config configs/experiment_name.yaml
    ```
4.  **Analyze Results**:
    - Open MLflow: `mlflow ui`
    - Compare metrics (Accuracy, Loss, AUC)
    - Check artifacts (Confusion Matrix, ROC Curves)

### Hyperparameter Tuning
**Trigger**: Optimizing model performance.

1.  **Edit Search Space**: Modify `get_hyperparameter_grid()` in `src/tune.py`.
2.  **Execute Search**:
    ```bash
    python src/tune.py --epochs 10
    ```
3.  **Select Best Model**: Check MLflow for the run with highest validation accuracy.

---

## 3. Deployment Procedures

### Deploying to Staging (Local Docker)
**Trigger**: Verifying model before production.

1.  **Build Image**:
    ```bash
    docker build -t plant-disease-classifier:staging -f serve/Dockerfile .
    ```
2.  **Run Container**:
    ```bash
    docker run -p 8000:8000 plant-disease-classifier:staging
    ```
3.  **Test Endpoint**:
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@test_image.jpg"
    ```

### Deploying to Production (Kubernetes)
**Trigger**: Promoting a validated model.

1.  **Update Model URI**: Edit `deploy/kubernetes/deployment.yaml` env var `MODEL_URI` to point to the new model version (e.g., `models:/custom_cnn_production/v2`).
2.  **Apply Changes**:
    ```bash
    kubectl apply -f deploy/kubernetes/deployment.yaml
    ```
3.  **Verify Rollout**:
    ```bash
    kubectl rollout status deployment/plant-disease-classifier
    ```

### Rollback Procedure
**Trigger**: Production issue detected immediately after deployment.

1.  **Execute Rollback**:
    ```bash
    kubectl rollout undo deployment/plant-disease-classifier
    ```

---

## 4. Monitoring & Incident Response

### Checking System Health
- **API Health**: `GET /health` should return `200 OK`.
- **Metrics**: Check Prometheus/Grafana dashboards for:
  - Request Latency (p95 > 500ms is critical)
  - Error Rate (> 1% is warning)
  - Prediction Confidence (Sudden drop indicates drift)

### Handling Data Drift
**Trigger**: Drift alert from Evidently AI.

1.  **Investigate Report**: Open the generated HTML drift report.
2.  **Identify Drifted Features**: Check which image properties or metadata drifted.
3.  **Action**:
    - If valid drift (seasonality): Trigger **Retraining Pipeline**.
    - If invalid drift (sensor error): Fix data source.

---

## 5. Maintenance Tasks

### Weekly Retraining
**Automated via Airflow**:
- DAG: `plant_disease_retraining`
- Schedule: Weekly (Sunday)

**Manual Trigger**:
```bash
airflow dags trigger plant_disease_retraining
```

### Cleaning Up Artifacts
**Trigger**: Disk space low.

1.  **Clean MLflow**:
    ```bash
    mlflow gc --experiment-ids <id>
    ```
2.  **Prune Docker Images**:
    ```bash
    docker system prune -a
    ```

---

## Troubleshooting Guide

| Issue | Possible Cause | Resolution |
|-------|----------------|------------|
| **Training OOM** | Batch size too large | Reduce `batch_size` in config |
| **DVC Pull Fail** | Auth credentials missing | Check cloud provider credentials |
| **API 503 Error** | Model not loaded | Check `MODEL_URI` and MLflow server |
| **High Latency** | CPU throttling | Increase K8s resource limits |
