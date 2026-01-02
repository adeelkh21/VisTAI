# Unified inference script for both classification and segmentation with Grad-CAM
# Usage: python inference.py --task classification --image path/to/image.jpg
#        python inference.py --task segmentation --image path/to/image.jpg

import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import json
import numpy as np

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from classification.models.efficientnet_classifier import create_efficientnet_classifier
from segmentation.mobilenetv2_unet import create_segmentation_model
from common.gradcam import apply_gradcam_classification, apply_gradcam_segmentation, GradCAM
from common.utils import load_checkpoint


def load_classification_model(checkpoint_path, device):
    with open(project_root / 'label_encoding.json', 'r') as f:
        label_info = json.load(f)
    
    model = create_efficientnet_classifier(num_classes=label_info['num_classes'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, label_info


def load_segmentation_model(checkpoint_path, device):
    model = create_segmentation_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path, image_size=224):
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    original_image = image.resize((image_size, image_size))
    
    return image_tensor, original_image


def inference_classification(image_path, checkpoint_path, device):
    # Load model
    model, label_info = load_classification_model(checkpoint_path, device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, image_size=224)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        class_logits = model(image_tensor)
        probabilities = torch.softmax(class_logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
    
    predicted_label = label_info['idx_to_label'][str(predicted_class_idx)]
    
    # Generate Grad-CAM
    target_layer = dict([*model.named_modules()])['features.8']
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor, target_class=predicted_class_idx)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(original_image)
    axes[1].imshow(cam, alpha=0.5, cmap='jet')
    axes[1].set_title(f'Grad-CAM\nPredicted: {predicted_label}\nConfidence: {confidence:.2%}')
    axes[1].axis('off')
    
    # Top-3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    top3_labels = [label_info['idx_to_label'][str(idx.item())] for idx in top3_indices]
    top3_text = '\n'.join([f"{label}: {prob:.2%}" for label, prob in zip(top3_labels, top3_probs)])
    
    axes[2].text(0.1, 0.5, f"Top 3 Predictions:\n\n{top3_text}", 
                fontsize=12, verticalalignment='center')
    axes[2].axis('off')
    
    plt.tight_layout()
    output_path = project_root / 'classification' / 'inference_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved result to {output_path}")
    plt.show()
    
    return predicted_label, confidence


def inference_segmentation(image_path, checkpoint_path, device):
    # Load model
    model = load_segmentation_model(checkpoint_path, device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path, image_size=384)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        pred_logits = model(image_tensor)
        pred_mask = torch.sigmoid(pred_logits) > 0.5
    
    # Generate Grad-CAM
    target_layer = model.encoder4
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    pred_mask_np = pred_mask[0, 0].cpu().numpy()
    axes[1].imshow(pred_mask_np, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    axes[2].imshow(original_image)
    axes[2].imshow(pred_mask_np, alpha=0.4, cmap='Reds')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    axes[3].imshow(original_image)
    axes[3].imshow(cam, alpha=0.5, cmap='jet')
    axes[3].set_title('Grad-CAM')
    axes[3].axis('off')
    
    plt.tight_layout()
    output_path = project_root / 'segmentation' / 'inference_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved result to {output_path}")
    plt.show()
    
    tumor_pixels = pred_mask_np.sum()
    total_pixels = pred_mask_np.size
    tumor_percentage = 100 * tumor_pixels / total_pixels
    
    return tumor_percentage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                       choices=['classification', 'segmentation'],
                       help='Task to perform')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: best checkpoint)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Default checkpoint paths
    if args.checkpoint is None:
        if args.task == 'classification':
            args.checkpoint = project_root / 'classification' / 'outputs' / 'checkpoint_best.pth'
        else:
            args.checkpoint = project_root / 'segmentation' / 'outputs' / 'checkpoint_best.pth'
    
    # Run inference
    if args.task == 'classification':
        label, confidence = inference_classification(args.image, args.checkpoint, device)
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    else:
        tumor_pct = inference_segmentation(args.image, args.checkpoint, device)
        print(f"\nTumor coverage: {tumor_pct:.2f}%")


if __name__ == '__main__':
    main()
