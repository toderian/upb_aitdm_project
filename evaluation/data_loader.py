"""
Data Loader for Evaluation - Works with extracted dataset
COVIDx CXR-4 Dataset (extracted to dataset/archive/)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


class COVIDxDataset(Dataset):
    """
    Dataset for COVIDx CXR-4 from extracted directory.
    """
    def __init__(self, data_dir: str, split: str = 'test', source_filter=None, transform=None):
        """
        Args:
            data_dir: Path to extracted archive directory (e.g., 'dataset/archive')
            split: One of 'train', 'val', 'test'
            source_filter: Optional list of sources to filter by
            transform: Image transforms
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # Image folder and txt file paths
        self.img_folder = os.path.join(data_dir, split)
        txt_file = os.path.join(data_dir, f'{split}.txt')

        # Read metadata
        self.df = pd.read_csv(
            txt_file,
            sep=' ',
            header=None,
            names=['pid', 'filename', 'label', 'source']
        )

        # Filter by source if specified
        if source_filter:
            if isinstance(source_filter, list):
                self.df = self.df[self.df['source'].isin(source_filter)]
            else:
                self.df = self.df[self.df['source'] == source_filter]
            self.df = self.df.reset_index(drop=True)

        # Label mapping: positive=1 (COVID), negative=0 (non-COVID)
        self.label_map = {'positive': 1, 'negative': 0}

        print(f"[{split.upper()}] Loaded {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        label_str = row['label']

        # Load image
        img_path = os.path.join(self.img_folder, img_name)

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return placeholder
            return torch.zeros((3, 224, 224)), torch.tensor(0)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        return img, label

    def get_class_distribution(self):
        """Get class distribution in the dataset."""
        labels = self.df['label'].map(self.label_map).values
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(['Negative', 'Positive'], counts))


def get_standard_transform():
    """Standard transforms for evaluation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_dataloader(data_dir: str = 'dataset/archive', batch_size: int = BATCH_SIZE):
    """
    Create test data loader.

    Args:
        data_dir: Path to extracted dataset directory
        batch_size: Batch size for DataLoader
    """
    transform = get_standard_transform()

    dataset = COVIDxDataset(
        data_dir=data_dir,
        split='test',
        source_filter=None,  # Use all sources
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


def get_validation_dataloader(data_dir: str = 'dataset/archive', batch_size: int = BATCH_SIZE):
    """Create validation data loader."""
    transform = get_standard_transform()

    dataset = COVIDxDataset(
        data_dir=data_dir,
        split='val',
        source_filter=None,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the data loader
    data_dir = "dataset/archive"

    if os.path.exists(data_dir):
        print("Testing data loader...")

        test_loader = get_test_dataloader(data_dir)
        print(f"Test loader batches: {len(test_loader)}")

        # Get one batch
        images, labels = next(iter(test_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels[:10]}")
    else:
        print(f"Data directory not found: {data_dir}")
