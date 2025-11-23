#!/usr/bin/env bash
# Quick start script for MLOps plant disease classification

echo "=== MLOps Plant Disease Classification Setup ==="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python --version

# Install dependencies
echo ""
echo "[2/5] Installing dependencies..."
pip install -r requirements.txt --upgrade --quiet

# Initialize DVC
echo ""
echo "[3/5] Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC initialized successfully"
else
    echo "DVC already initialized"
fi

# Create directories
echo ""
echo "[4/5] Setting up directories..."
mkdir -p data/raw data/processed checkpoints logs mlruns

# Run tests
echo ""
echo "[5/5] Running tests..."
pytest tests/unit/ -v --tb=short || echo "Tests require dataset to be present"

echo ""
echo "=== Setup Complete ===" 
echo ""
echo "Next steps:"
echo "1. Configure DVC remote storage: dvc remote add -d storage <url>"
echo "2. Add your dataset to data/raw/Dataset"
echo "3. Train your model: python src/train.py"
echo "4. View MLflow UI: mlflow ui"
echo ""
