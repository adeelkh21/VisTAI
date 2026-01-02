# VisTAI - Bone Tumor X-Ray Detection System

AI-powered system for bone tumor classification and segmentation from X-ray images with multimodal chat interface.

## Features

- **Classification**: EfficientNet-based tumor classification (9 classes)
- **Segmentation**: MobileNetV2-UNet for tumor region detection
- **Multimodal Chat**: Interactive medical assistant with CLIP and LLM integration
- **Web Interface**: Streamlit-based demo application

## Quick Start

```bash
# Install dependencies
pip install -r streamlit_requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

- `classification/` - Tumor classification module
- `segmentation/` - Tumor segmentation module  
- `multimodal/` - LLM/VLM pipeline for conversational interface
- `common/` - Shared utilities and visualization tools

## Tumor Classes

Giant Cell Tumor, Multiple Osteochondromas, Osteochondroma, Osteofibroma, Osteosarcoma, Other Benign Tumors, Other Malignant Tumors, Simple Bone Cyst, Synovial Osteochondroma
