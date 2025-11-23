import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

from src.models.cnn_model import COVIDxCNN
from src.data.dataset import COVIDxClientDataset
from src.data.preprocessing import get_train_transforms, get_test_transforms

class COVIDxClient(fl.client.NumPyClient):
    """
    Flower client for federated learning on COVIDx dataset.
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data."""
        self.set_parameters(parameters)

        local_epochs = config.get("local_epochs", 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total
        }

        print(f"  Client training - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}")

        return self.get_parameters(), total, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / total

        print(f"  Client evaluation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy, "loss": avg_loss}


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Client {args.client_id} using device: {device}")

    # Load client data
    data_path = f"data/federated/client_{args.client_id}_data.npz"
    print(f"Loading data from {data_path}")

    train_dataset = COVIDxClientDataset(data_path, 'train', get_train_transforms())
    val_dataset = COVIDxClientDataset(data_path, 'test', get_test_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"Client {args.client_id}: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Model
    model = COVIDxCNN(num_classes=4, pretrained=True)

    # Create client
    client = COVIDxClient(model, train_loader, val_loader, device)

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)