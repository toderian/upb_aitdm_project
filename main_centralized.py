# main_centralized.py
import argparse
from src.models.cnn_model import COVIDxCNN
from src.models.train_centralized import Trainer
from src.data.dataset import COVIDxDataset
from src.data.preprocessing import get_train_transforms, get_test_transforms

import torch

from torch.utils.data import DataLoader

def main(args):
    # Config
    config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'class_weights': [2.0, 1.0, 1.0, 1.5]  # Adjust based on class imbalance
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_dataset = COVIDxDataset('data/processed', 'train', get_train_transforms())
    val_dataset = COVIDxDataset('data/processed', 'test', get_test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = COVIDxCNN(num_classes=4, pretrained=True)

    # Train
    trainer = Trainer(model, device, config)
    history = trainer.train(train_loader, val_loader, epochs=args.epochs)

    # Save
    trainer.save_history('results/stage1/centralized_history.json')
    print(f"Training complete. Best model saved to models/best_centralized.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)