"""
Visual inference script - shows model predictions with images
"""

import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import json
import yaml
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# Get test transform
transform = get_classification_transforms(phase='test')

# Load test CSV
test_csv = project_root / CONFIG['data']['augmented_test_csv']
test_df = pd.read_csv(test_csv)

# Select 10 images (stratified)
print(f"\nTotal test images: {len(test_df)}")
print("Selecting 10 images for inference...")

sample_images = []
for class_name in test_df['labels'].unique():
    class_samples = test_df[test_df['labels'] == class_name].sample(n=min(2, len(test_df[test_df['labels'] == class_name])))
    sample_images.append(class_samples)

sample_df = pd.concat(sample_images).head(10).reset_index(drop=True)

print("\nRunning inference...")

# Run inference
correct = 0
results = []

with torch.no_grad():
    for idx, row in sample_df.iterrows():
        image_path = row['image_path']
        true_label = row['labels']
        
        # Load original image for display
        original_image = Image.open(image_path).convert('RGB')
        
        # Transform for model
        image_tensor = transform(original_image).unsqueeze(0).to(device)
        
        # Forward pass
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        predicted_label = idx_to_label[predicted_idx.item()]
        confidence_val = confidence.item()
        
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
        
        results.append({
            'image': original_image,
            'image_name': Path(image_path).name,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence_val,
            'correct': is_correct
        })

# Create visualization
print(f"\nCreating visualization...")
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle(f'Classification Results: {correct}/10 Correct ({correct*10}%)', 
             fontsize=20, fontweight='bold', y=0.98)

for idx, (ax, result) in enumerate(zip(axes.flat, results)):
    # Display image
    ax.imshow(result['image'])
    ax.axis('off')
    
    # Prepare title
    true_label = result['true_label']
    pred_label = result['predicted_label']
    confidence = result['confidence']
    is_correct = result['correct']
    
    # Color code: green for correct, red for wrong
    title_color = 'green' if is_correct else 'red'
    status = '✓ CORRECT' if is_correct else '✗ WRONG'
    
    # Title with multiple lines
    title = f"{status}\n"
    title += f"True: {true_label}\n"
    title += f"Pred: {pred_label}\n"
    title += f"Confidence: {confidence:.1%}"
    
    ax.set_title(title, fontsize=10, color=title_color, fontweight='bold', pad=10)
    
    # Add border
    rect = patches.Rectangle((0, 0), result['image'].width, result['image'].height,
                             linewidth=4, edgecolor=title_color, facecolor='none')
    ax.add_patch(rect)

plt.tight_layout()

# Save figure
output_path = project_root / 'classification/inference_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# Print summary
print("\n" + "="*80)
print(f"ACCURACY: {correct}/10 = {correct*10}%")
print("="*80)

print("\nDetailed Results:")
for idx, result in enumerate(results, 1):
    status = "✓" if result['correct'] else "✗"
    print(f"{idx:2d}. {status} {result['image_name']:20s} | True: {result['true_label']:25s} | "
          f"Pred: {result['predicted_label']:25s} | Conf: {result['confidence']:.1%}")

# Show per-class accuracy
print("\nPer-Class Results:")
for class_name in sorted(set(r['true_label'] for r in results)):
    class_results = [r for r in results if r['true_label'] == class_name]
    class_correct = sum(r['correct'] for r in class_results)
    print(f"  {class_name:30s}: {class_correct}/{len(class_results)} correct")

print("\nVisualization window opening...")
plt.show()
