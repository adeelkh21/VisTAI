"""
MobileNetV2 Inference Service
Loads the trained MobileNetV2 model and provides classification predictions.
Simple and fast lightweight model for Jetson deployment.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import json
from pathlib import Path

from app.config import Settings

# ── Label map ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "giant cell tumor",
    "multiple osteochondromas",
    "osteochondroma",
    "osteofibroma",
    "osteosarcoma",
    "other bt",
    "other mt",
    "simple bone cyst",
    "synovial osteochondroma",
]


class MobileNetV2Model(nn.Module):
    """MobileNetV2 model wrapper with custom classification head."""
    
    def __init__(self, num_classes: int = 9):
        super().__init__()
        from torchvision.models import mobilenet_v2
        
        # Load pretrained MobileNetV2
        self.backbone = mobilenet_v2(weights='DEFAULT')
        
        # Replace classification head (match training architecture)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MobileNetInferenceService:
    """Service for MobileNetV2 classification inference."""
    
    def __init__(self, settings: Settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.settings = settings
        self.model = None
        self.class_names = CLASS_NAMES
        self.num_classes = len(CLASS_NAMES)
        
        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained MobileNetV2 model."""
        # The model path should point to the outputs directory
        # Try multiple possible locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "BTXRD" / "mobilenet" / "outputs" / "best_model.pth",
            Path(__file__).parent.parent.parent.parent / "BTXRD" / "mobilenet" / "outputs" / "best_model.pth",
            Path.home() / "mobilenet" / "best_model.pth",
            Path("/data/deployment/models/mobilenet_best.pth"),
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"MobileNetV2 model not found. Checked: {possible_paths}"
            )
        
        print(f"Loading MobileNetV2 from: {model_path}")
        
        # Create model and load weights
        self.model = MobileNetV2Model(self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ MobileNetV2 loaded successfully on {self.device}")
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict tumor class from X-ray image.
        
        Args:
            image: Input image as numpy array (H, W, C) with values 0-255
            
        Returns:
            dict with keys:
                - class: Predicted class name (str)
                - confidence: Confidence score (float 0-1)
                - probabilities: Dict of all class probabilities
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Normalize to 0-255
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            pil_image = image
        
        # Convert to RGB if grayscale
        if pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        # Preprocess
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            confidence = probabilities[predicted_idx].item()
        
        # Build response
        prob_dict = {
            self.class_names[i]: float(probabilities[i].item())
            for i in range(self.num_classes)
        }
        
        return {
            "class": self.class_names[predicted_idx],
            "confidence": confidence,
            "probabilities": prob_dict,
        }
    
    def predict_batch(self, images: list) -> list:
        """
        Predict multiple images efficiently.
        
        Args:
            images: List of numpy arrays
            
        Returns:
            List of prediction dicts
        """
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
