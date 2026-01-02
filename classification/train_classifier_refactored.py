# Classification training script for EfficientNet-B0
# Multi-class bone tumor classification

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'common'))

from classification.models.efficientnet_classifier import create_efficientnet_classifier
from classification.datasets.classification_dataset import create_classification_dataloaders
from common.losses import FocalLoss
from common.metrics import compute_classification_metrics
from common.utils import save_checkpoint, AverageMeter


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    
    epoch_loss = AverageMeter()
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Train')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            class_logits = model(images)
            loss = criterion(class_logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Compute accuracy
        _, predicted_classes = torch.max(class_logits, 1)
        total += labels.size(0)
        correct += (predicted_classes == labels).sum().item()
        epoch_loss.update(loss.item(), labels.size(0))
        
        pbar.set_postfix({
            'loss': f'{epoch_loss.avg:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return epoch_loss.avg, 100.0 * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    
    val_loss = AverageMeter()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Val'):
            images = images.to(device)
            labels = labels.to(device)
            
            class_logits = model(images)
            loss = criterion(class_logits, labels)
            
            _, predicted_classes = torch.max(class_logits, 1)
            
            val_loss.update(loss.item(), labels.size(0))
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_classes.cpu().numpy())
    
    # Compute metrics
    metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics['loss'] = val_loss.avg
    
    return metrics


def train_classification(num_epochs=None, batch_size=None, learning_rate=None):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Config
    config = {
        'num_classes': 9,
        'image_size': 224,
        'num_epochs': num_epochs or 30,
        'batch_size': batch_size or 32,
        'learning_rate': learning_rate or 0.001,
        'weight_decay': 1e-4,
    }
    
    # Load label encoding
    with open(project_root / 'label_encoding.json', 'r') as f:
        label_info = json.load(f)
    
    # Create dataloaders
    train_csv = project_root / 'augmented_classification_data' / 'augmented_train.csv'
    val_csv = project_root / 'augmented_classification_data' / 'augmented_val.csv'
    test_csv = project_root / 'augmented_classification_data' / 'augmented_test.csv'
    
    dataloaders = create_classification_dataloaders(
        str(train_csv), str(val_csv), str(test_csv),
        label_info['label_to_idx'],
        batch_size=config['batch_size'],
        num_workers=4
    )
    
    # Create model
    model = create_efficientnet_classifier(num_classes=config['num_classes'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=None, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=5)
    scaler = GradScaler('cuda')
    
    # Output directory
    output_dir = project_root / 'classification' / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, scaler
        )
        
        # Validate
        val_metrics = validate(model, dataloaders['val'], criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val   - F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Save checkpoints
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1']},
            output_dir / 'checkpoint_latest.pth'
        )
        
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'val_accuracy': val_metrics['accuracy'], 'val_f1': val_metrics['f1']},
                output_dir / 'checkpoint_best.pth'
            )
            print(f"✓ Best model saved! Accuracy: {val_metrics['accuracy']:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
            
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch} epochs")
            break
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Classification training complete. Best Accuracy: {best_acc:.2f}%")
    return best_acc


if __name__ == '__main__':
    train_classification()
