"""
MobileNet Inference API – POST /api/mobilenet/predict
Fast lightweight inference endpoint for tumor classification.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
import io

from app import main as app_main

router = APIRouter(prefix="/mobilenet", tags=["mobilenet"])

mobilenet_service = None  # Will be initialized from main


class MobileNetPrediction(BaseModel):
    """Response model for MobileNet prediction."""
    class_name: str
    confidence: float
    probabilities: dict


@router.post("/predict", response_model=MobileNetPrediction)
async def predict_mobilenet(file: UploadFile = File(...)):
    """
    Run MobileNetV2 inference on an uploaded X-ray image.
    
    Returns:
    - class_name: Predicted tumor type
    - confidence: Confidence score (0-1)
    - probabilities: All class probabilities
    
    **Example:**
    ```
    curl -X POST -F "file=@image.png" http://localhost:8000/api/mobilenet/predict
    ```
    """
    if app_main.mobilenet_service is None:
        raise HTTPException(
            status_code=503, 
            detail="MobileNetV2 model is still loading. Try again shortly."
        )
    
    try:
        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_array = np.array(image)
        
        # Run inference
        result = app_main.mobilenet_service.predict(image_array)
        
        return MobileNetPrediction(
            class_name=result["class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    is_loaded = app_main.mobilenet_service is not None
    return {
        "status": "ok" if is_loaded else "loading",
        "model_loaded": is_loaded,
        "model": "MobileNetV2"
    }


@router.get("/info")
async def model_info():
    """Get model metadata."""
    if app_main.mobilenet_service is None:
        return {"status": "loading"}
    
    service = app_main.mobilenet_service
    return {
        "model": "MobileNetV2",
        "input_size": 224,
        "num_classes": service.num_classes,
        "classes": service.class_names,
        "device": str(service.device),
    }
