# Main training entry point for both classification and segmentation models

import argparse
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from classification.train_classifier_refactored import train_classification
from segmentation.train_segmentation_refactored import train_segmentation


def run_combined_inference(num_images=10):
    # Run inference on same 10 images for both models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("RUNNING COMBINED INFERENCE ON 10 TEST IMAGES")
    print("="*80)
    
    # Load models
    from segmentation.mobilenetv2_unet import create_segmentation_model
    from classification.models.efficientnet_classifier import create_efficientnet_classifier
    from common.gradcam import GradCAM
    import json
    
    # Load segmentation model
    seg_model = create_segmentation_model(pretrained=False)
    seg_checkpoint = torch.load(project_root / 'segmentation/outputs/checkpoint_best.pth', map_location=device)
    seg_model.load_state_dict(seg_checkpoint['model_state_dict'])
    seg_model = seg_model.to(device)
    seg_model.eval()
    
    # Load classification model
    with open(project_root / 'label_encoding.json', 'r') as f:
        label_info = json.load(f)
    
    cls_model = create_efficientnet_classifier(num_classes=label_info['num_classes'])
    cls_checkpoint = torch.load(project_root / 'classification/outputs/checkpoint_best.pth', map_location=device)
    cls_model.load_state_dict(cls_checkpoint['model_state_dict'])
    cls_model = cls_model.to(device)
    cls_model.eval()
    
    # Load test data (find common images in both test sets)
    seg_df = pd.read_csv(project_root / 'segmentation_test.csv')
    cls_df = pd.read_csv(project_root / 'augmented_classification_data/augmented_test.csv')
    
    # Use segmentation test images
    test_images = seg_df.head(num_images)
    
    # Create output directory
    output_dir = project_root / 'combined_inference_results'
    output_dir.mkdir(exist_ok=True)
    
    # Transforms
    seg_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Setup Grad-CAM
    seg_gradcam = GradCAM(seg_model, seg_model.encoder4)
    cls_gradcam = GradCAM(cls_model, dict([*cls_model.named_modules()])['backbone.features.8'])
    
    for idx, row in test_images.iterrows():
        img_path = row['image_path']
        mask_path = row['mask_path']
        true_label = row['labels']
        
        print(f"\n[{idx+1}/{num_images}] Processing: {Path(img_path).name}")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Segmentation inference
        seg_img = image.resize((384, 384))
        seg_tensor = seg_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            seg_logits = seg_model(seg_tensor)
            seg_mask = (torch.sigmoid(seg_logits) > 0.5).float()[0, 0].cpu().numpy()
        
        seg_cam = seg_gradcam.generate(seg_tensor)
        
        # Classification inference
        cls_img = image.resize((224, 224))
        cls_tensor = cls_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cls_logits = cls_model(cls_tensor)
            cls_probs = torch.softmax(cls_logits, dim=1)
            pred_class_idx = torch.argmax(cls_probs, dim=1).item()
            confidence = cls_probs[0, pred_class_idx].item()
        
        pred_label = label_info['idx_to_label'][str(pred_class_idx)]
        cls_cam = cls_gradcam.generate(cls_tensor, target_class=pred_class_idx)
        
        # Load ground truth mask
        true_mask = Image.open(mask_path).convert('L').resize((384, 384), Image.NEAREST)
        true_mask_np = (np.array(true_mask) / 255.0 > 0.1).astype(np.float32)
        
        # Compute segmentation metrics
        intersection = (seg_mask * true_mask_np).sum()
        union = seg_mask.sum() + true_mask_np.sum()
        dice = (2.0 * intersection) / (union + 1e-8)
        
        # Visualize
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
        
        # Row 1: Segmentation
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(seg_img)
        ax1.set_title(f'Original\n{true_label}', fontsize=10)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(true_mask_np, cmap='gray')
        ax2.set_title('Ground Truth Mask', fontsize=10)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(seg_mask, cmap='gray')
        ax3.set_title(f'Predicted Mask\nDice: {dice:.3f}', fontsize=10)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(seg_img)
        ax4.imshow(seg_mask, alpha=0.4, cmap='Reds')
        ax4.set_title('Segmentation Overlay', fontsize=10)
        ax4.axis('off')
        
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(seg_img)
        ax5.imshow(seg_cam, alpha=0.5, cmap='jet')
        ax5.set_title('Seg Grad-CAM', fontsize=10)
        ax5.axis('off')
        
        # Row 2: Classification
        ax6 = fig.add_subplot(gs[1, 0])
        ax6.imshow(cls_img)
        ax6.set_title(f'Original\n{true_label}', fontsize=10)
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 1])
        ax7.imshow(cls_img)
        ax7.imshow(cls_cam, alpha=0.5, cmap='jet')
        ax7.set_title('Classification Grad-CAM', fontsize=10)
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 2])
        match = "✓" if pred_label == true_label else "✗"
        ax8.text(0.5, 0.5, f'{match} Prediction:\n{pred_label}\n\nConfidence:\n{confidence:.1%}', 
                ha='center', va='center', fontsize=12, 
                color='green' if match == "✓" else 'red',
                weight='bold')
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        # Top 3 predictions
        top3_probs, top3_indices = torch.topk(cls_probs[0], min(3, label_info['num_classes']))
        top3_labels = [label_info['idx_to_label'][str(idx.item())] for idx in top3_indices]
        top3_text = 'Top 3:\n\n' + '\n'.join([f'{i+1}. {label}: {prob:.1%}' 
                                               for i, (label, prob) in enumerate(zip(top3_labels, top3_probs))])
        
        ax9 = fig.add_subplot(gs[1, 3:])
        ax9.text(0.1, 0.5, top3_text, ha='left', va='center', fontsize=11, family='monospace')
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        
        plt.suptitle(f'Combined Analysis - Image {idx+1}', fontsize=14, weight='bold')
        
        # Save
        filename = Path(img_path).stem
        save_path = output_dir / f'{filename}_combined.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Segmentation: Dice={dice:.3f}")
        print(f"  Classification: {pred_label} ({confidence:.1%}) - {'Correct' if match=='✓' else 'Wrong'}")
        print(f"  Saved: {save_path.name}")
    
    print(f"\n✓ All results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['classification', 'segmentation', 'both'], 
                       default='both', help='Which task to train')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs from config')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--inference_only', action='store_true', help='Skip training, only run inference')
    args = parser.parse_args()
    
    if not args.inference_only:
        if args.task in ['classification', 'both']:
            print("\n" + "="*80)
            print("TRAINING CLASSIFICATION MODEL (EfficientNet-B0)")
            print("="*80)
            train_classification(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        
        if args.task in ['segmentation', 'both']:
            print("\n" + "="*80)
            print("TRAINING SEGMENTATION MODEL (MobileNetV2-UNet)")
            print("="*80)
            train_segmentation(
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
        
        print("\n" + "="*80)
        print("✓ TRAINING COMPLETE")
        print("="*80)
    
    # Run combined inference
    run_combined_inference(num_images=10)


if __name__ == '__main__':
    main()
