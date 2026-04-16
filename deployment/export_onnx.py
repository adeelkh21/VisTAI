#!/usr/bin/env python3
"""
BTXRD ONNX Export Script

Run this script on your PC BEFORE deploying to Jetson.
It exports the trained .pth student models to ONNX format.
The .onnx files are then copied to the Jetson where trtexec
converts them to .trt TensorRT engines.

Usage:
  python export_onnx.py

Output files (saved to same folder as source .pth files):
  classification_student.onnx
  segmentation_student.onnx
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _load_env_file(path: Path) -> None:
    """Lightweight .env loader to avoid requiring extra dependencies."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load env files in precedence order (existing process env wins).
_load_env_file(SCRIPT_DIR / ".env")
_load_env_file(SCRIPT_DIR / ".env.local")

# Check for onnxruntime
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnx and onnxruntime are required.")
    print("Install with: pip install onnx onnxruntime")
    sys.exit(1)

# Import model classes from services
sys.path.insert(0, str(Path(__file__).parent / "btxrd-backend"))
from app.services.classification_service import _ConvNeXtTinyStudent
from app.services.segmentation_service import _SegFormerB2Student


def get_model_dir() -> Path:
    """Get model directory from environment or default."""
    default_model_dir = PROJECT_ROOT / "BTXRD" / "combined_inference" / "models"
    model_dir = os.environ.get("MODEL_DIR", str(default_model_dir))
    return Path(model_dir).expanduser().resolve()


def export_classification(model_dir: Path) -> bool:
    """Export classification model to ONNX."""
    model_path = model_dir / "classification_student.pth"
    onnx_path = model_path.with_suffix(".onnx")

    if not model_path.exists():
        print(f"ERROR: Classification model not found: {model_path}")
        print(f"Set MODEL_DIR environment variable to the correct path.")
        print(f"Current MODEL_DIR: {model_dir}")
        return False

    print(f"Loading classification model from {model_path}...")

    # Load model exactly as service does
    model = _ConvNeXtTinyStudent(num_classes=9)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {
        (k.replace("backbone.", "") if k.startswith("backbone.") else k): v
        for k, v in sd.items()
    }
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Create dummy input matching inference preprocessing
    # classification_service.py: T.Resize(416), T.CenterCrop(384)
    # Input shape: (1, 3, 384, 384)
    dummy_input = torch.randn(1, 3, 384, 384)

    print(f"Exporting to ONNX (opset_version=17)...")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"Exported: {onnx_path}")

    # Validation
    print("Validating ONNX model...")
    try:
        onnx.checker.check_model(onnx.load(str(onnx_path)))

        # Run inference comparison
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

        if np.allclose(pytorch_output, onnx_output, atol=1e-4):
            print("PASSED: ONNX output matches PyTorch (atol=1e-4)")
            return True
        else:
            max_diff = np.abs(pytorch_output - onnx_output).max()
            print(f"FAILED: Max difference = {max_diff}")
            return False

    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        return False


def export_segmentation(model_dir: Path) -> bool:
    """Export segmentation model to ONNX."""
    model_path = model_dir / "segmentation_student.pth"
    onnx_path = model_path.with_suffix(".onnx")

    if not model_path.exists():
        print(f"ERROR: Segmentation model not found: {model_path}")
        print(f"Set MODEL_DIR environment variable to the correct path.")
        print(f"Current MODEL_DIR: {model_dir}")
        return False

    print(f"Loading segmentation model from {model_path}...")

    # Load model exactly as service does
    model = _SegFormerB2Student(num_classes=1, image_size=224)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Create dummy input matching inference preprocessing
    # segmentation_service.py: resize to 224x224
    # Input shape: (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting to ONNX (opset_version=17)...")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"Exported: {onnx_path}")

    # Validation
    print("Validating ONNX model...")
    try:
        onnx.checker.check_model(onnx.load(str(onnx_path)))

        # Run inference comparison
        with torch.no_grad():
            # _SegFormerB2Student.forward returns logits tensor directly.
            pytorch_output = model(dummy_input).numpy()

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

        if np.allclose(pytorch_output, onnx_output, atol=1e-4):
            print("PASSED: ONNX output matches PyTorch (atol=1e-4)")
            return True
        else:
            max_diff = np.abs(pytorch_output - onnx_output).max()
            print(f"FAILED: Max difference = {max_diff}")
            return False

    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("BTXRD ONNX Export Script")
    print("=" * 60)

    model_dir = get_model_dir()
    print(f"Model directory: {model_dir}")
    print()

    cls_success = export_classification(model_dir)
    print()
    seg_success = export_segmentation(model_dir)

    print()
    print("=" * 60)
    if cls_success and seg_success:
        print("SUCCESS: Both models exported and validated!")
        print(f"Output files:")
        print(f"  - {model_dir / 'classification_student.onnx'}")
        print(f"  - {model_dir / 'segmentation_student.onnx'}")
    else:
        print("FAILED: One or more exports failed.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
