"""
Create train/val/test splits for segmentation task (80/10/10).
Uses stratified sampling based on tumor class labels.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_segmentation_splits():
    """Create stratified train/val/test splits for segmentation."""
    
    # Load the full dataset CSV
    csv_path = r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD\classification_labels.csv"
    df = pd.read_csv(csv_path)
    
    # Filter only samples that have masks
    df = df[df['has_mask'] == True].copy()
    print(f"Total samples with masks: {len(df)}")
    
    # Check class distribution
    print("\nClass distribution:")
    print(df['labels'].value_counts())
    
    # First split: 80% train, 20% temp (which will be split into val and test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['labels']
    )
    
    # Second split: Split temp into 50/50 for val and test (10% each of original)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['labels']
    )
    
    print(f"\n📊 Split Summary:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify class distribution in each split
    print("\n📈 Class Distribution per Split:")
    print("\nTrain:")
    print(train_df['labels'].value_counts().sort_index())
    print("\nValidation:")
    print(val_df['labels'].value_counts().sort_index())
    print("\nTest:")
    print(test_df['labels'].value_counts().sort_index())
    
    # Save splits
    base_dir = r"c:\Users\Nauman\Desktop\vistai\FYP\BTXRD"
    
    train_path = os.path.join(base_dir, "segmentation_train.csv")
    val_path = os.path.join(base_dir, "segmentation_val.csv")
    test_path = os.path.join(base_dir, "segmentation_test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Splits saved:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print(f"  Test:  {test_path}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    print("=" * 80)
    print("CREATING SEGMENTATION DATASET SPLITS")
    print("=" * 80)
    print()
    
    train_df, val_df, test_df = create_segmentation_splits()
    
    print("\n" + "=" * 80)
    print("SPLITS CREATED SUCCESSFULLY!")
    print("=" * 80)
