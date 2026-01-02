# Bone Tumor Classification Pipeline

Independent classification pipeline using EfficientNet-B0 backbone with two-phase training strategy.

## Features

- **Model**: EfficientNet-B0 (5.3M parameters, ImageNet pretrained)
- **Training Strategy**: 
  - Phase 1 (10 epochs): Frozen backbone, train only classifier head
  - Phase 2 (40 epochs): Full end-to-end fine-tuning
- **Loss**: Focal Loss (γ=2.0) with class weighting for imbalance
- **Data**: Augmented dataset with 8,958 training images (6x original)
- **Optimizations**: EMA, mixed precision, gradient clipping

## Directory Structure

```
classification/
├── models/
│   ├── efficientnet_classifier.py   # EfficientNet-B0 classifier
│   └── __init__.py
├── datasets/
│   ├── classification_dataset.py    # Data loader
│   └── __init__.py
├── configs/
│   └── efficientnet_config.yaml    # Training configuration
├── train_classifier.py              # Main training script
├── test_classifier.py               # Evaluation script (TODO)
└── README.md                        # This file
```

## Quick Start

### 1. Prepare Data

The pipeline can use either:
- **Augmented dataset** (recommended): `augmented_classification_data/` (8,958 train images)
- **Original dataset**: `segmentation_train.csv` (1,493 images)

Set `use_augmented: true/false` in `configs/efficientnet_config.yaml`

### 2. Configure Training

Edit `configs/efficientnet_config.yaml`:
```yaml
training:
  phase1_epochs: 10   # Frozen backbone
  phase2_epochs: 40   # Fine-tuning
  batch_size: 32
  
  optimizer:
    lr_phase1: 0.001  # Higher LR for frozen
    lr_phase2: 0.0001 # Lower LR for fine-tuning
```

### 3. Train Model

```bash
cd c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\classification
python train_classifier.py
```

Training will:
1. Load augmented data (8,958 images)
2. Compute class weights for imbalanced classes
3. **Phase 1**: Train classifier head with frozen backbone (10 epochs)
4. **Phase 2**: Fine-tune entire network (40 epochs)
5. Save best model based on validation accuracy

### 4. Monitor Training

Output files saved to `classification/outputs/efficientnet_b0/`:
```
outputs/
├── checkpoints/
│   ├── phase1_best.pth      # Best model from phase 1
│   └── best_model.pth        # Best overall model
├── logs/
├── training_history.npy      # Training metrics
└── training_curves.png       # Loss/accuracy plots
```

## Model Architecture

```
Input: [B, 3, 224, 224]
    ↓
EfficientNet-B0 Encoder (pretrained ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Fully Connected (1280 → 9 classes)
    ↓
Output: Class logits [B, 9]
```

### Freeze/Unfreeze

```python
from models.efficientnet_classifier import create_efficientnet_classifier

model = create_efficientnet_classifier(num_classes=9)

# Phase 1: Freeze backbone
model.freeze_backbone()  # Only classifier trains

# Phase 2: Unfreeze all
model.unfreeze_backbone()  # Full fine-tuning
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `phase1_epochs` | 10 | Epochs with frozen backbone |
| `phase2_epochs` | 40 | Epochs for fine-tuning |
| `lr_phase1` | 0.001 | Learning rate (frozen) |
| `lr_phase2` | 0.0001 | Learning rate (fine-tuning) |
| `dropout` | 0.3 | Dropout before final FC |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |

### Class Weights

Automatically computed from training data to handle class imbalance:
```python
class_weight[i] = total_samples / (num_classes * class_count[i])
```

## Expected Performance

| Metric | Target |
|--------|--------|
| Training Accuracy | ~95% |
| Validation Accuracy | **>75%** |
| Test Accuracy | **>70%** |

*Significant improvement expected with 6x augmented data*

## Training Tips

1. **Use augmented data** for best results (set `use_augmented: true`)
2. **Phase 1 is crucial**: Ensure classifier head learns well before unfreezing
3. **Monitor validation accuracy**: Early stopping prevents overfitting
4. **Adjust learning rates**: Increase if loss plateaus, decrease if unstable

## Troubleshooting

**Out of Memory?**
- Reduce `batch_size` to 16 or 24
- Reduce `num_workers` to 2

**Slow training?**
- Increase `num_workers` to 6-8
- Ensure augmented data is on SSD

**Overfitting?**
- Increase dropout to 0.4-0.5
- Add more regularization (weight_decay)
- Enable early stopping

## Next Steps

After training:
1. Run `test_classifier.py` for comprehensive evaluation
2. Generate confusion matrix and per-class metrics
3. Test on unseen data
4. Use in unified inference pipeline with segmentation

## Citation

If you use this pipeline, please cite:
- EfficientNet: Tan & Le, ICML 2019
- Focal Loss: Lin et al., ICCV 2017
