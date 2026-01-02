"""
Segmentation Training Script with GradCAM support
"""

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import json

# Add modules to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from models.efficientunet import create_efficientunet
from datasets.segmentation_dataset import create_segmentation_dataloaders
from common.losses import CombinedSegmentationLoss
from common.metrics import compute_dice_score, compute_iou, compute_pixel_accuracy
from common.utils import set_seed, AverageMeter, save_checkpoint
from common.visualization import visualize_segmentation, plot_training_curves

# Load config
with open(project_root / 'segmentation/configs/efficientunet_config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    dice_scores = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        if CONFIG['training'].get('use_mixed_precision', False) and device.type == 'cuda':
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        
        if CONFIG['training'].get('use_mixed_precision', False) and device.type == 'cuda':
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
            optimizer.step()
        
        # Metrics
        with torch.no_grad():
            dice = compute_dice_score(outputs, masks)
        
        losses.update(loss.item(), images.size(0))
        dice_scores.update(dice.item(), images.size(0))
        
        pbar.set_postfix({'loss': losses.avg, 'dice': dice_scores.avg})
    
    return losses.avg, dice_scores.avg
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if CONFIG['training']['gradient_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            dice = compute_dice_score(preds.float(), masks)
        
        losses.update(loss.item(), images.size(0))
        dice_scores.update(dice if isinstance(dice, float) else dice.item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'dice': f'{dice_scores.avg:.4f}'
        })
    
    return losses.avg, dice_scores.avg


def validate(model, dataloader, criterion, device, epoch, output_dir=None):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    dice_scores = AverageMeter()
    iou_scores = AverageMeter()
    pixel_accs = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Validation")):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Predictions
            preds = torch.sigmoid(outputs) > 0.5
            
            # Compute metrics per sample
            for i in range(images.size(0)):
                dice = compute_dice_score(preds[i:i+1].float(), masks[i:i+1])
                iou = compute_iou(preds[i:i+1].float(), masks[i:i+1])
                pixel_acc = compute_pixel_accuracy(preds[i:i+1].float(), masks[i:i+1])
                
                dice_val = dice if isinstance(dice, float) else dice.item()
                iou_val = iou if isinstance(iou, float) else iou.item()
                pixel_val = pixel_acc if isinstance(pixel_acc, float) else pixel_acc.item()
                
                dice_scores.update(dice_val)
                iou_scores.update(iou_val)
                pixel_accs.update(pixel_val)
            
            losses.update(loss.item(), images.size(0))
            
            # Visualize first batch
            if batch_idx == 0 and output_dir is not None:
                viz_dir = output_dir / 'visualizations'
                viz_dir.mkdir(exist_ok=True)
                
                for i in range(min(4, images.size(0))):
                    visualize_segmentation(
                        images[i].cpu(),
                        masks[i].cpu(),
                        preds[i].cpu(),
                        save_path=viz_dir / f'epoch_{epoch}_sample_{i}.png'
                    )
    
    print(f"\nValidation - Loss: {losses.avg:.4f}, Dice: {dice_scores.avg:.4f}, "
          f"IoU: {iou_scores.avg:.4f}, Pixel Acc: {pixel_accs.avg:.4f}")
    
    return losses.avg, dice_scores.avg, iou_scores.avg, pixel_accs.avg


def main():
    # Setup
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(CONFIG['seed'])
    
    # Create output directory
    base_path = project_root.parent
    output_dir = base_path / CONFIG['output']['save_dir']
    checkpoint_dir = output_dir / CONFIG['output']['checkpoint_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    
    # Data paths
    train_csv = base_path / CONFIG['data']['train_csv']
    val_csv = base_path / CONFIG['data']['val_csv']
    test_csv = base_path / CONFIG['data']['test_csv']
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    loaders = create_segmentation_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=CONFIG['training']['batch_size'],
        num_workers=CONFIG['training']['num_workers']
    )
    
    # Create model
    print("\nCreating model...")
    model = create_efficientunet(
        num_classes=CONFIG['model']['num_classes'],
        pretrained=CONFIG['model']['pretrained']
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedSegmentationLoss(
        dice_weight=CONFIG['training']['loss']['dice_weight'],
        bce_weight=CONFIG['training']['loss']['bce_weight']
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['training']['optimizer']['lr'],
        weight_decay=CONFIG['training']['optimizer'].get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=CONFIG['training']['scheduler']['mode'],
        factor=CONFIG['training']['scheduler']['factor'],
        patience=CONFIG['training']['scheduler']['patience'],
        min_lr=CONFIG['training']['scheduler']['min_lr'],
        verbose=True
    )
    
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'val_pixel_acc': []
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    print("\n" + "="*80)
    print(f"Starting training for {CONFIG['training']['epochs']} epochs")
    print("="*80)
    
    # Training loop
    for epoch in range(1, CONFIG['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['training']['epochs']}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device, scaler, epoch
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_pixel_acc = validate_epoch(
            model, loaders['val'], criterion, device, epoch, 
            save_dir=output_dir / CONFIG['output']['visualization_dir']
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_pixel_acc = validate(
            model, loaders['val'], criterion, device, epoch, output_dir
        )
        
        # Update scheduler
        scheduler.step(val_dice)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_pixel_acc'].append(val_pixel_acc)
        
        # Save checkpoint if best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                {'val_dice': val_dice, 'val_loss': val_loss},
                checkpoint_dir / 'best_model.pth'
            )
            print(f"✓ New best model saved! Dice: {val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if epoch % CONFIG['output']['save_frequency'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_dice': val_dice, 'val_loss': val_loss},
                checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            )
        
        # Early stopping
        if CONFIG['training']['early_stopping']['enabled']:
            if patience_counter >= CONFIG['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        # Save training curves
        np.save(output_dir / 'training_history.npy', history)
        plot_training_curves(
            history,
            save_path=output_dir / 'training_curves.png',
            title='Segmentation Training Curves'
        )
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
