# main_centralized.py
import argparse
from model_side.models.cnn_model import COVIDxCNN
from model_side.models.train_centralized import Trainer
# from model_side.data.preprocessing import get_train_transforms, get_test_transforms
from model_side.data.data_loader_enhanced import *
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

IMG_SIZE = (224, 224)
transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

def main(args):
    # Config
    config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'class_weights': [1.0, 2.0]  # Adjust weights for 2 classes (negative, positive)
    }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # train_dataset = COVIDxZipDataset('archive.zip', 'train.txt', transform=transform)
    # val_dataset = COVIDxZipDataset('archive.zip', 'test.txt', transform=transform)
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    train_loader = get_federated_client(client_id=1, batch_size=args.batch_size)
    val_loader = get_client_validation(client_id=1, batch_size=args.batch_size)
    # Model
    model = COVIDxCNN(num_classes=2, pretrained=True)

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