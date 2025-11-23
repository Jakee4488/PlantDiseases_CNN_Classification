"""
FastAPI application for model serving.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from typing import Dict, Any, List
import mlflow.tensorflow
from pydantic import BaseModel
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases (Healthy, Powdery, Rust)",
    version="1.0.0"
)

# Constants
IMG_SIZE = (256, 256)
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']
MODEL_URI = os.getenv('MODEL_URI', 'models:/custom_cnn_production/Production')

# Global model variable
model = None

# Metrics
PREDICTION_COUNT = Counter(
    'prediction_count', 
    'Number of predictions', 
    ['status', 'model_version']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 
    'Time spent processing prediction'
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time: float


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    try:
        print(f"Loading model from {MODEL_URI}...")
        # In a real scenario, we would load from MLflow
        # For local testing without MLflow server, we might load a local path
        # model = mlflow.tensorflow.load_model(MODEL_URI)
        
        # Fallback for development/testing if MLflow not available
        if os.path.exists('checkpoints/custom_cnn_best.h5'):
             model = tf.keras.models.load_model('checkpoints/custom_cnn_best.h5')
        else:
            print("Warning: No model found. API will fail on predict.")
            
    except Exception as e:
        print(f"Error loading model: {e}")


def preprocess_image(image_data: bytes) -> np.ndarray:
    """Preprocess image for model inference."""
    image = Image.open(io.BytesIO(image_data))
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    
    # Normalize if needed (model expects 0-1)
    image_array = image_array.astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_uri": MODEL_URI}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from image file.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG supported.")
        
        # Read and preprocess
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        # Inference
        predictions = model.predict(processed_image)
        
        # Process results
        predicted_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        predicted_class = CLASS_NAMES[predicted_idx]
        
        probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, predictions[0])
        }
        
        duration = time.time() - start_time
        
        # Update metrics
        PREDICTION_COUNT.labels(status='success', model_version='v1').inc()
        PREDICTION_LATENCY.observe(duration)
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "processing_time": duration
        }
        
    except Exception as e:
        PREDICTION_COUNT.labels(status='error', model_version='v1').inc()
        raise HTTPException(status_code=500, detail=str(e))
