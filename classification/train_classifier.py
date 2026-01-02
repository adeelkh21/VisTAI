"""
Classification Training Script
==============================
Two-phase training: frozen backbone → full fine-tuning
"""

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys

# Add modules to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from models.efficientnet_classifier import create_efficientnet_classifier
from datasets.classification_dataset import create_classification_dataloaders
from common.losses import FocalLoss
from common.metrics import compute_classification_metrics
from common.utils import set_seed, EMA, AverageMeter, save_checkpoint
from common.visualization import plot_training_curves, plot_confusion_matrix

# Load config
with open(project_root / 'classification/configs/efficientnet_config.yaml', 'r') as f:
    CONFIG = yaml.safe_load(f)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, ema=None, phase='frozen'):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Training ({phase})")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision forward
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['training']['gradient_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        if ema:
            ema.update()
        
        # Metrics
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        losses.update(loss.item(), labels.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return {
        'loss': losses.avg,
        'accuracy': 100.0 * correct / total
    }


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    
    losses = AverageMeter()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            losses.update(loss.item(), labels.size(0))
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    metrics = compute_classification_metrics(all_labels, all_preds)
    
    return {
        'loss': losses.avg,
        'accuracy': metrics['accuracy'] * 100,
        'confusion_matrix': metrics['confusion_matrix']
    }


def main():
    """Main training function."""
    
    print("="*70)
    print("BONE TUMOR CLASSIFICATION TRAINING")
    print("="*70)
    
    # Setup
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(CONFIG['seed'])
    
    base_path = project_root.parent / "BTXRD"
    output_dir = base_path / CONFIG['output']['save_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load label encoding
    with open(base_path / CONFIG['data']['label_encoding'], 'r') as f:
        encoding = json.load(f)
        label_to_idx = encoding['label_to_idx']
        idx_to_label = {int(k): v for k, v in encoding['idx_to_label'].items()}
        num_classes = encoding['num_classes']
    
    print(f"Number of classes: {num_classes}")
    
    # Data paths
    if CONFIG['data']['use_augmented']:
        train_csv = base_path / CONFIG['data']['augmented_train_csv']
        val_csv = base_path / CONFIG['data']['augmented_val_csv']
        test_csv = base_path / CONFIG['data']['augmented_test_csv']
        print("Using AUGMENTED dataset")
    else:
        train_csv = base_path / CONFIG['data']['original_train_csv']
        val_csv = base_path / CONFIG['data']['original_val_csv']
        test_csv = base_path / CONFIG['data']['original_test_csv']
        print("Using ORIGINAL dataset")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    loaders = create_classification_dataloaders(
        train_csv, val_csv, test_csv,
        label_to_idx,
        batch_size=CONFIG['training']['batch_size'],
        num_workers=CONFIG['training']['num_workers']
    )
    
    # Compute class weights
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    class_counts = train_df['labels'].map(label_to_idx).value_counts().sort_index()
    total_samples = len(train_df)
    class_weights = torch.FloatTensor([
        total_samples / (num_classes * class_counts[i]) 
        for i in range(num_classes)
    ]).to(device)
    
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # Create model
    print("\nCreating model...")
    model = create_efficientnet_classifier(
        num_classes=num_classes,
        pretrained=CONFIG['model']['pretrained'],
        dropout=CONFIG['model']['dropout']
    )
    model = model.to(device)
    
    # Loss function
    criterion = FocalLoss(
        alpha=class_weights if CONFIG['training']['loss']['use_class_weights'] else None,
        gamma=CONFIG['training']['loss']['gamma']
    )
    
    # Training utilities
    scaler = GradScaler()
    ema = EMA(model, decay=CONFIG['training']['ema_decay']) if CONFIG['training']['use_ema'] else None
    
    # Training history
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "="*70)
    print("PHASE 1: Training with Frozen Backbone")
    print("="*70)
    
    # Phase 1: Frozen backbone
    model.freeze_backbone()
    
    optimizer_phase1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['training']['optimizer']['lr_phase1'],
        weight_decay=CONFIG['training']['optimizer']['weight_decay']
    )
    
    scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase1,
        mode=CONFIG['training']['scheduler']['mode'],
        factor=CONFIG['training']['scheduler']['factor'],
        patience=CONFIG['training']['scheduler']['patience']
    )
    
    for epoch in range(CONFIG['training']['phase1_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['training']['phase1_epochs']}")
        
        train_metrics = train_one_epoch(
            model, loaders['train'], criterion, 
            optimizer_phase1, device, scaler, ema, phase='frozen'
        )
        
        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, loaders['val'], criterion, device)
        if ema:
            ema.restore()
        
        scheduler_phase1.step(val_metrics['accuracy'])
        
        print(f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if ema:
                ema.apply_shadow()
            save_checkpoint(
                model, optimizer_phase1, scheduler_phase1,
                epoch + 1, val_metrics,
                output_dir / 'checkpoints' / 'phase1_best.pth',
                is_best=True
            )
            if ema:
                ema.restore()
    
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning (Unfrozen Backbone)")
    print("="*70)
    
    # Phase 2: Unfreeze and fine-tune
    model.unfreeze_backbone()
    
    optimizer_phase2 = optim.AdamW(
        model.parameters(),
        lr=CONFIG['training']['optimizer']['lr_phase2'],
        weight_decay=CONFIG['training']['optimizer']['weight_decay']
    )
    
    scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_phase2,
        mode=CONFIG['training']['scheduler']['mode'],
        factor=CONFIG['training']['scheduler']['factor'],
        patience=CONFIG['training']['scheduler']['patience']
    )
    
    for epoch in range(CONFIG['training']['phase2_epochs']):
        print(f"\nEpoch {CONFIG['training']['phase1_epochs'] + epoch+1}/{CONFIG['training']['total_epochs']}")
        
        train_metrics = train_one_epoch(
            model, loaders['train'], criterion,
            optimizer_phase2, device, scaler, ema, phase='unfrozen'
        )
        
        if ema:
            ema.apply_shadow()
        val_metrics = validate(model, loaders['val'], criterion, device)
        if ema:
            ema.restore()
        
        scheduler_phase2.step(val_metrics['accuracy'])
        
        print(f"Train - Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            if ema:
                ema.apply_shadow()
            save_checkpoint(
                model, optimizer_phase2, scheduler_phase2,
                CONFIG['training']['phase1_epochs'] + epoch + 1,
                val_metrics,
                output_dir / 'checkpoints' / 'best_model.pth',
                is_best=True
            )
            if ema:
                ema.restore()
            print(f"★ NEW BEST MODEL (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
    
    # Save training history
    np.save(output_dir / 'training_history.npy', history)
    plot_training_curves(history, save_path=output_dir / 'training_curves.png')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_dir / 'checkpoints' / 'best_model.pth'}")


if __name__ == "__main__":
    main()
