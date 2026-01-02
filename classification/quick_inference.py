"""
Quick inference script to test model on 10 images
"""

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
import yaml
from pathlib import Path
import sys

# Add modules to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from models.efficientnet_classifier import create_efficientnet_classifier
from common.transforms import get_classification_transforms

# Load config
with open(project_root / 'classification/configs/efficientnet_config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load label encoding
with open(project_root / CONFIG['data']['label_encoding'], 'r') as f:
    encoding = json.load(f)
    label_to_idx = encoding['label_to_idx']
    idx_to_label = {int(k): v for k, v in encoding['idx_to_label'].items()}
    num_classes = encoding['num_classes']

# Load model
print("\nLoading best model...")
model = create_efficientnet_classifier(num_classes=num_classes, pretrained=False)
checkpoint_path = project_root / 'classification/outputs/efficientnet_b0/checkpoints/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
if 'best_val_accuracy' in checkpoint:
    print(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
if 'val_accuracy' in checkpoint:
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.4f}")

# Get test transform
transform = get_classification_transforms(phase='test')

# Load test CSV
test_csv = project_root / CONFIG['data']['augmented_test_csv']
test_df = pd.read_csv(test_csv)

# Select 10 random images (stratified by class if possible)
print(f"\nTotal test images: {len(test_df)}")
print("\nSelecting 10 images for inference...")

# Try to get at least one from each class
sample_images = []
for class_name in test_df['labels'].unique():
    class_samples = test_df[test_df['labels'] == class_name].sample(n=min(2, len(test_df[test_df['labels'] == class_name])))
    sample_images.append(class_samples)

# Combine and take first 10
sample_df = pd.concat(sample_images).head(10).reset_index(drop=True)

print("\nRunning inference on 10 test images:")
print("=" * 80)

correct = 0
results = []

with torch.no_grad():
    for idx, row in sample_df.iterrows():
        # Load image
        image_path = row['image_path']
        true_label = row['labels']
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Forward pass
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_label = idx_to_label[predicted_idx.item()]
        confidence_val = confidence.item()
        
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probs[0], 3)
        top3_labels = [(idx_to_label[idx.item()], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]
        
        results.append({
            'image': Path(image_path).name,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence_val,
            'correct': is_correct,
            'top3': top3_labels
        })
        
        # Print result
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        print(f"\n{idx+1}. Image: {Path(image_path).name}")
        print(f"   True:      {true_label}")
        print(f"   Predicted: {predicted_label} ({confidence_val:.2%}) {status}")
        print(f"   Top 3: ", end="")
        for label, prob in top3_labels:
            print(f"{label} ({prob:.2%})", end="  ")
        print()

print("\n" + "=" * 80)
print(f"\nAccuracy on 10 samples: {correct}/10 = {correct/10:.1%}")

# Class-wise breakdown
print("\nPer-class results:")
for class_name in sample_df['labels'].unique():
    class_results = [r for r in results if r['true_label'] == class_name]
    class_correct = sum(r['correct'] for r in class_results)
    print(f"  {class_name}: {class_correct}/{len(class_results)} correct")

print("\nConfidence statistics:")
confidences = [r['confidence'] for r in results]
correct_confidences = [r['confidence'] for r in results if r['correct']]
wrong_confidences = [r['confidence'] for r in results if not r['correct']]

print(f"  Average confidence (all): {sum(confidences)/len(confidences):.2%}")
if correct_confidences:
    print(f"  Average confidence (correct): {sum(correct_confidences)/len(correct_confidences):.2%}")
if wrong_confidences:
    print(f"  Average confidence (wrong): {sum(wrong_confidences)/len(wrong_confidences):.2%}")

print("\nDone!")
