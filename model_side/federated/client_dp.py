"""
Flower client with local Differential Privacy using Opacus.
Implements (ε, δ)-differential privacy for federated learning.
"""
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple
from tqdm import tqdm

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from model_side.models.cnn_model import COVIDxCNN, SimpleCNN
from model_side.data.data_loader_enhanced import get_federated_client, get_client_validation


class COVIDxDPClient(fl.client.NumPyClient):
    """
    Flower client with local Differential Privacy.
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 target_epsilon, target_delta=1e-5, max_grad_norm=1.0):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        
        # Make model DP-compatible using ModuleValidator
        self.model = ModuleValidator.fix(model)
        if not ModuleValidator.is_valid(self.model):
            raise ValueError("Model not compatible with Opacus after fix attempt")
        self.model = self.model.to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.current_epsilon = 0

    def get_parameters(self, config=None) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays."""
        # Handle wrapped model from PrivacyEngine
        if hasattr(self.model, '_module'):
            state_dict = self.model._module.state_dict()
        else:
            state_dict = self.model.state_dict()
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays."""
        # Get the actual model (unwrapped if needed)
        if hasattr(self.model, '_module'):
            model_to_update = self.model._module
        else:
            model_to_update = self.model
            
        params_dict = zip(model_to_update.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model_to_update.load_state_dict(state_dict, strict=True)

    def _setup_dp_training(self, local_epochs):
        """Set up DP training for this round."""
        # Create fresh optimizer for each round
        if hasattr(self.model, '_module'):
            params = self.model._module.parameters()
        else:
            params = self.model.parameters()
            
        optimizer = torch.optim.SGD(params, lr=1e-3, momentum=0.9)
        
        # Create PrivacyEngine
        privacy_engine = PrivacyEngine()
        
        # Make private
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=local_epochs,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm
        )
        
        return model, optimizer, train_loader, privacy_engine

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data with Differential Privacy."""
        self.set_parameters(parameters)
        
        local_epochs = config.get("local_epochs", 1)
        
        # Set up DP for this round
        model, optimizer, train_loader, privacy_engine = self._setup_dp_training(local_epochs)
        
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for epoch in range(local_epochs):
            for images, labels in tqdm(train_loader, desc=f"DP Training Epoch {epoch+1}/{local_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Get spent epsilon
        self.current_epsilon = privacy_engine.get_epsilon(delta=self.target_delta)
        
        # Update model reference (unwrap if needed)
        self.model = model._module if hasattr(model, '_module') else model
        
        metrics = {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "epsilon": float(self.current_epsilon)
        }
        
        print(f"  DP Client - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, ε: {self.current_epsilon:.2f}")
        
        return self.get_parameters(), total, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model on local validation data."""
        self.set_parameters(parameters)
        
        if hasattr(self.model, '_module'):
            model = self.model._module
        else:
            model = self.model
            
        model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        print(f"  DP Client eval - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
        
        return avg_loss, total, {"accuracy": accuracy, "loss": avg_loss}


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DP Client {args.client_id} using device: {device}")
    print(f"Target epsilon: {args.target_epsilon}")
    
    # Load client data using enhanced loader
    train_loader = get_federated_client(args.client_id, batch_size=args.batch_size)
    val_loader = get_client_validation(args.client_id, batch_size=args.batch_size)
    
    print(f"Client {args.client_id}: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    
    # Model - use SimpleCNN for DP (smaller, more compatible)
    model = SimpleCNN(num_classes=args.num_classes)
    
    # Create DP client
    client = COVIDxDPClient(
        model, train_loader, val_loader, device,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm
    )
    
    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower DP Client")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--target_epsilon", type=float, default=5.0, 
                        help="Target epsilon for differential privacy")
    parser.add_argument("--target_delta", type=float, default=1e-5,
                        help="Target delta for differential privacy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    args = parser.parse_args()
    main(args)
