# Training script for MobileNetV2-UNet segmentation model
# Binary tumor segmentation with Dice + BCE loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from segmentation.mobilenetv2_unet import create_segmentation_model, count_parameters
from segmentation.segmentation_dataset import create_segmentation_dataloaders
from common.utils import save_checkpoint, load_checkpoint, AverageMeter
from common.metrics import compute_dice_score, compute_iou


class SegmentationLossDiceBCE(nn.Module):
    # Combined Dice + BCE loss for binary segmentation
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_logits, target_mask):
        # BCE loss
        bce_loss = self.bce(pred_logits, target_mask)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred_logits)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target_mask.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    
    epoch_loss = AverageMeter()
    epoch_dice = AverageMeter()
    epoch_iou = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Train')
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            pred_logits = model(images)
            loss = criterion(pred_logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        with torch.no_grad():
            dice = compute_dice_score(pred_logits, masks)
            iou = compute_iou(pred_logits, masks)
        
        epoch_loss.update(loss.item(), images.size(0))
        epoch_dice.update(dice, images.size(0))
        epoch_iou.update(iou, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{epoch_loss.avg:.4f}',
            'dice': f'{epoch_dice.avg:.4f}',
            'iou': f'{epoch_iou.avg:.4f}'
        })
    
    return epoch_loss.avg, epoch_dice.avg, epoch_iou.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    
    val_loss = AverageMeter()
    val_dice = AverageMeter()
    val_iou = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Val')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            pred_logits = model(images)
            loss = criterion(pred_logits, masks)
            
            dice = compute_dice_score(pred_logits, masks)
            iou = compute_iou(pred_logits, masks)
            
            val_loss.update(loss.item(), images.size(0))
            val_dice.update(dice, images.size(0))
            val_iou.update(iou, images.size(0))
            
            pbar.set_postfix({
                'loss': f'{val_loss.avg:.4f}',
                'dice': f'{val_dice.avg:.4f}',
                'iou': f'{val_iou.avg:.4f}'
            })
    
    return val_loss.avg, val_dice.avg, val_iou.avg


def train_segmentation(num_epochs=None, batch_size=None, learning_rate=None):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Config
    config = {
        'image_size': 384,
        'num_epochs': num_epochs or 25,
        'batch_size': batch_size or 8,
        'learning_rate': learning_rate or 0.001,
        'weight_decay': 1e-5,
    }
    
    # Create dataloaders
    train_csv = project_root / 'segmentation_train.csv'
    val_csv = project_root / 'segmentation_val.csv'
    test_csv = project_root / 'segmentation_test.csv'
    
    dataloaders = create_segmentation_dataloaders(
        str(train_csv), str(val_csv), str(test_csv),
        batch_size=config['batch_size'],
        image_size=config['image_size']
    )
    
    # Create model
    model = create_segmentation_model(pretrained=True)
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = SegmentationLossDiceBCE(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                     factor=0.5, patience=5)
    scaler = GradScaler('cuda')
    
    # Output directory
    output_dir = project_root / 'segmentation' / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_dice = 0.0
    patience_counter = 0
    early_stop_patience = 15
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, scaler, device
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, dataloaders['val'], criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_dice)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save checkpoints
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            {'val_dice': val_dice, 'val_iou': val_iou},
            output_dir / 'checkpoint_latest.pth'
        )
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_dice': val_dice, 'val_iou': val_iou},
                output_dir / 'checkpoint_best.pth'
            )
            print(f"✓ Best model saved! Dice: {val_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Segmentation training complete. Best Dice: {best_dice:.4f}")
    return best_dice


if __name__ == '__main__':
    train_segmentation()
