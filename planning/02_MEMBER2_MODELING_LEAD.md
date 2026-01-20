# Member 2: Modeling & Privacy Lead
## Detailed Working Plan

---

## Role Summary

| Attribute | Details |
|-----------|---------|
| **Primary Role** | Modeling & Privacy Lead |
| **Main Responsibilities** | Model architecture, centralized training, Flower FL implementation, Differential Privacy |
| **Key Tools** | PyTorch, Flower (flwr), Opacus |
| **Trust Dimension (Stage 2)** | Privacy (Differential Privacy) |

---

## Stage 1: Baseline Development (Week 1-2)

### Task 1.1: Research Model Architectures
**Duration:** Day 1-2
**Priority:** HIGH

#### Actionables:
1. **Survey CNN architectures for medical imaging**

   | Model | Parameters | Pros | Cons |
   |-------|------------|------|------|
   | ResNet18 | 11M | Well-tested, good baseline | Larger for FL |
   | ResNet34 | 21M | Better accuracy | Communication overhead |
   | EfficientNet-B0 | 5M | Efficient, good accuracy | More complex |
   | DenseNet121 | 7M | Feature reuse | Memory intensive |
   | MobileNetV2 | 3.4M | Very lightweight | Lower accuracy |
   | Custom CNN | <1M | Minimal overhead | May underperform |

2. **Consider FL-specific requirements**
   - Model size affects communication cost
   - Recommendation: Start with ResNet18, optionally compare with smaller model
   - Consider: Can we use pretrained weights? (ImageNet)

3. **Review related work**
   - Check COVID-Net architecture (specifically designed for COVIDx)
   - Review FL papers on medical imaging
   - Document 3-5 relevant papers

#### Deliverables:
- [ ] Architecture comparison table
- [ ] Selected model with justification
- [ ] Related work notes (for report)

---

### Task 1.2: Implement Base Model Architecture
**Duration:** Day 2-3
**Priority:** HIGH

#### Actionables:
1. **Create model module** `src/models/cnn_model.py`

2. **Implement CNN Model**
   ```python
   import torch
   import torch.nn as nn
   from torchvision import models

   class COVIDxCNN(nn.Module):
       """
       CNN for COVIDx classification.
       Based on ResNet18 with modified final layer.
       """
       def __init__(self, num_classes=4, pretrained=True):
           super(COVIDxCNN, self).__init__()

           # Load pretrained ResNet18
           self.backbone = models.resnet18(pretrained=pretrained)

           # Modify first conv for grayscale (if needed)
           # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

           # Modify final layer for our classes
           in_features = self.backbone.fc.in_features
           self.backbone.fc = nn.Sequential(
               nn.Dropout(0.5),
               nn.Linear(in_features, num_classes)
           )

       def forward(self, x):
           return self.backbone(x)

       def get_num_parameters(self):
           return sum(p.numel() for p in self.parameters() if p.requires_grad)


   class SimpleCNN(nn.Module):
       """
       Lightweight CNN for faster FL experiments.
       """
       def __init__(self, num_classes=4):
           super(SimpleCNN, self).__init__()
           self.features = nn.Sequential(
               nn.Conv2d(3, 32, kernel_size=3, padding=1),
               nn.BatchNorm2d(32),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2, 2),

               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(64),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2, 2),

               nn.Conv2d(64, 128, kernel_size=3, padding=1),
               nn.BatchNorm2d(128),
               nn.ReLU(inplace=True),
               nn.MaxPool2d(2, 2),

               nn.Conv2d(128, 256, kernel_size=3, padding=1),
               nn.BatchNorm2d(256),
               nn.ReLU(inplace=True),
               nn.AdaptiveAvgPool2d((1, 1))
           )
           self.classifier = nn.Sequential(
               nn.Flatten(),
               nn.Dropout(0.5),
               nn.Linear(256, num_classes)
           )

       def forward(self, x):
           x = self.features(x)
           x = self.classifier(x)
           return x
   ```

3. **Test model creation**
   ```python
   # Test script
   model = COVIDxCNN(num_classes=4)
   print(f"Model parameters: {model.get_num_parameters():,}")

   # Test forward pass
   x = torch.randn(4, 3, 224, 224)
   output = model(x)
   print(f"Output shape: {output.shape}")  # Should be [4, 4]
   ```

#### Deliverables:
- [ ] `src/models/cnn_model.py`
- [ ] Model parameter count documented

---

### Task 1.3: Implement Centralized Training
**Duration:** Day 3-5
**Priority:** HIGH (partially blocked by Member 1's data)

#### Actionables:
1. **Create training module** `src/models/train_centralized.py`

2. **Implement training loop**
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   import json
   from tqdm import tqdm

   class Trainer:
       def __init__(self, model, device, config):
           self.model = model.to(device)
           self.device = device
           self.config = config

           # Loss function with class weights for imbalance
           class_weights = torch.tensor(config.get('class_weights', [1.0, 1.0, 1.0, 1.0]))
           self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

           # Optimizer
           self.optimizer = optim.Adam(
               model.parameters(),
               lr=config.get('learning_rate', 1e-4),
               weight_decay=config.get('weight_decay', 1e-5)
           )

           # Learning rate scheduler
           self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
               self.optimizer, mode='min', factor=0.5, patience=3
           )

           self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

       def train_epoch(self, train_loader):
           self.model.train()
           total_loss = 0
           correct = 0
           total = 0

           for images, labels in tqdm(train_loader, desc="Training"):
               images, labels = images.to(self.device), labels.to(self.device)

               self.optimizer.zero_grad()
               outputs = self.model(images)
               loss = self.criterion(outputs, labels)
               loss.backward()
               self.optimizer.step()

               total_loss += loss.item() * images.size(0)
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()

           return total_loss / total, 100. * correct / total

       def validate(self, val_loader):
           self.model.eval()
           total_loss = 0
           correct = 0
           total = 0

           with torch.no_grad():
               for images, labels in val_loader:
                   images, labels = images.to(self.device), labels.to(self.device)
                   outputs = self.model(images)
                   loss = self.criterion(outputs, labels)

                   total_loss += loss.item() * images.size(0)
                   _, predicted = outputs.max(1)
                   total += labels.size(0)
                   correct += predicted.eq(labels).sum().item()

           return total_loss / total, 100. * correct / total

       def train(self, train_loader, val_loader, epochs):
           best_val_acc = 0

           for epoch in range(epochs):
               train_loss, train_acc = self.train_epoch(train_loader)
               val_loss, val_acc = self.validate(val_loader)

               self.scheduler.step(val_loss)

               self.history['train_loss'].append(train_loss)
               self.history['train_acc'].append(train_acc)
               self.history['val_loss'].append(val_loss)
               self.history['val_acc'].append(val_acc)

               print(f"Epoch {epoch+1}/{epochs}")
               print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
               print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

               # Save best model
               if val_acc > best_val_acc:
                   best_val_acc = val_acc
                   torch.save(self.model.state_dict(), 'models/best_centralized.pth')

           return self.history

       def save_history(self, path):
           with open(path, 'w') as f:
               json.dump(self.history, f)
   ```

3. **Create main training script**
   ```python
   # main_centralized.py
   import argparse
   from model_side.models.cnn_model import COVIDxCNN
   from model_side.models.train_centralized import Trainer
   from model_side.data.dataset import COVIDxDataset
   from model_side.data.preprocessing import get_train_transforms, get_test_transforms

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
   ```

4. **Create configuration file**
   ```yaml
   # configs/training_config.yaml
   centralized:
     epochs: 30
     batch_size: 32
     learning_rate: 0.0001
     weight_decay: 0.00001
     optimizer: adam
     scheduler: reduce_on_plateau

   model:
     architecture: resnet18
     pretrained: true
     num_classes: 4
     dropout: 0.5
   ```

#### Deliverables:
- [ ] `src/models/train_centralized.py`
- [ ] `configs/training_config.yaml`
- [ ] `models/best_centralized.pth`
- [ ] `results/stage1/centralized_history.json`

---

### Task 1.4: Implement Flower Server
**Duration:** Day 3-4 (parallel with centralized training setup)
**Priority:** HIGH

#### Actionables:
1. **Create server module** `src/federated/server.py`

2. **Implement Flower server**
   ```python
   import flwr as fl
   import argparse
   from typing import List, Tuple, Dict, Optional
   from flwr.common import Metrics

   def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
       """
       Aggregate metrics from multiple clients using weighted average.
       """
       # Collect accuracies and losses
       accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
       losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
       examples = [num_examples for num_examples, _ in metrics]

       return {
           "accuracy": sum(accuracies) / sum(examples),
           "loss": sum(losses) / sum(examples) if sum(losses) > 0 else 0
       }

   def get_on_fit_config(local_epochs: int):
       """
       Return function that creates fit config for each round.
       """
       def fit_config(server_round: int) -> Dict:
           return {
               "local_epochs": local_epochs,
               "current_round": server_round
           }
       return fit_config

   def main(args):
       # Define strategy
       strategy = fl.server.strategy.FedAvg(
           fraction_fit=1.0,  # Use all available clients for training
           fraction_evaluate=1.0,  # Use all clients for evaluation
           min_fit_clients=args.num_clients,
           min_evaluate_clients=args.num_clients,
           min_available_clients=args.num_clients,
           evaluate_metrics_aggregation_fn=weighted_average,
           fit_metrics_aggregation_fn=weighted_average,
           on_fit_config_fn=get_on_fit_config(args.local_epochs)
       )

       # Server config
       config = fl.server.ServerConfig(num_rounds=args.num_rounds)

       # Start server
       print(f"Starting Flower server on {args.server_address}")
       print(f"Waiting for {args.num_clients} clients...")
       print(f"Running {args.num_rounds} rounds with {args.local_epochs} local epochs each")

       fl.server.start_server(
           server_address=args.server_address,
           config=config,
           strategy=strategy
       )

   if __name__ == "__main__":
       parser = argparse.ArgumentParser(description="Flower Server")
       parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
       parser.add_argument("--num_rounds", type=int, default=10)
       parser.add_argument("--num_clients", type=int, default=3)
       parser.add_argument("--local_epochs", type=int, default=3)
       args = parser.parse_args()
       main(args)
   ```

3. **Test server startup**
   ```bash
   python model_side/federated/server.py --num_rounds 5 --num_clients 3 --local_epochs 2
   ```

#### Deliverables:
- [ ] `src/federated/server.py`
- [ ] Server successfully starts and waits for clients

---

### Task 1.5: Implement Flower Client
**Duration:** Day 4-6
**Priority:** HIGH (blocked by Member 1's data splits)

#### Actionables:
1. **Create client module** `src/federated/client.py`

2. **Implement Flower client**
   ```python
   import flwr as fl
   import torch
   import torch.nn as nn
   import numpy as np
   import argparse
   from collections import OrderedDict
   from typing import Dict, List, Tuple

   from model_side.models.cnn_model import COVIDxCNN
   from model_side.data.dataset import COVIDxClientDataset
   from model_side.data.preprocessing import get_train_transforms, get_test_transforms

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
   ```

3. **Create run script for multiple clients**
   ```bash
   #!/bin/bash
   # run_fl_experiment.sh

   # Start server in background
   python model_side/federated/server.py --num_rounds 10 --num_clients 3 --local_epochs 3 &
   SERVER_PID=$!

   # Wait for server to start
   sleep 5

   # Start clients
   python model_side/federated/client.py --client_id 1 --server_address 0.0.0.0:8080 &
   python model_side/federated/client.py --client_id 2 --server_address 0.0.0.0:8080 &
   python model_side/federated/client.py --client_id 3 --server_address 0.0.0.0:8080 &

   # Wait for all processes
   wait
   ```

#### Deliverables:
- [ ] `src/federated/client.py`
- [ ] `scripts/run_fl_experiment.sh`
- [ ] Working FL setup with 3 clients

---

### Task 1.6: Run Federated Experiments
**Duration:** Day 6-7
**Priority:** HIGH

#### Actionables:
1. **Run baseline FL experiment**
   ```bash
   # Run with 10 rounds, 3 clients, 3 local epochs
   ./scripts/run_fl_experiment.sh
   ```

2. **Save federated model**
   ```python
   # Add to server.py - save final model
   # Or extract from client after training
   ```

3. **Log and visualize results**
   - Training loss per round
   - Accuracy per round
   - Per-client metrics

4. **Compare centralized vs federated**

   | Metric | Centralized | Federated |
   |--------|-------------|-----------|
   | Final Accuracy | X% | X% |
   | Final Loss | X | X |
   | Training Time | Xmin | Xmin |
   | Convergence Round | N/A | X |

5. **Experiment with different configurations**
   - Vary number of rounds (5, 10, 20)
   - Vary local epochs (1, 3, 5)
   - Document impact on convergence

#### Deliverables:
- [ ] `models/federated_final.pth`
- [ ] `results/stage1/federated_history.json`
- [ ] Comparison notebook/visualizations
- [ ] Configuration that works best

---

### Task 1.7: Stage 1 Report Contribution
**Duration:** Day 7 (Week 2 end)
**Priority:** HIGH

#### Sections to Write:
1. **Model Architecture** (0.5 page)
   - Architecture choice and justification
   - Model parameters count
   - Modifications for task

2. **Centralized Baseline** (0.5 page)
   - Training setup (optimizer, scheduler, epochs)
   - Results on test set

3. **Federated Learning Setup** (1 page)
   - Flower framework overview
   - Server configuration (FedAvg strategy)
   - Client implementation
   - Number of rounds, local epochs

4. **Preliminary Results** (0.5 page)
   - Federated training curves
   - Comparison with centralized baseline
   - Observed limitations

#### Deliverables:
- [ ] ~2.5 pages of Stage 1 report content

---

## Stage 2: Privacy with Differential Privacy (Week 3-5)

### Task 2.1: Understand Differential Privacy
**Duration:** Week 3, Day 1
**Priority:** HIGH

#### Actionables:
1. **Study DP fundamentals**
   - Definition of (ε, δ)-differential privacy
   - Privacy budget composition
   - Utility-privacy trade-off

2. **Understand DP-SGD**
   - Gradient clipping
   - Noise addition
   - Privacy accountant

3. **Review Opacus library**
   - Documentation: https://opacus.ai/
   - Tutorials and examples

#### Key Concepts:
```
ε (epsilon): Privacy budget - lower = more private, less accurate
δ (delta): Probability of privacy breach - typically 10^-5
Gradient clipping: Bound gradient norms to limit sensitivity
Noise multiplier: Scale of Gaussian noise added
```

---

### Task 2.2: Implement DP-SGD Training
**Duration:** Week 3, Day 2-4
**Priority:** HIGH

#### Actionables:
1. **Create DP training module** `src/privacy/dp_training.py`

2. **Implement DP training with Opacus**
   ```python
   import torch
   import torch.nn as nn
   from opacus import PrivacyEngine
   from opacus.validators import ModuleValidator
   from opacus.utils.batch_memory_manager import BatchMemoryManager
   import numpy as np

   class DPTrainer:
       """
       Trainer with Differential Privacy using Opacus.
       """
       def __init__(self, model, device, config):
           self.device = device
           self.config = config

           # Make model compatible with Opacus
           self.model = ModuleValidator.fix(model)
           if not ModuleValidator.is_valid(self.model):
               raise ValueError("Model not compatible with Opacus")

           self.model = self.model.to(device)
           self.criterion = nn.CrossEntropyLoss()
           self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

           self.privacy_engine = None
           self.history = {'train_loss': [], 'train_acc': [], 'epsilon': []}

       def attach_privacy_engine(self, train_loader, target_epsilon, target_delta, epochs, max_grad_norm):
           """
           Attach Opacus PrivacyEngine to the model and optimizer.
           """
           self.privacy_engine = PrivacyEngine()

           self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
               module=self.model,
               optimizer=self.optimizer,
               data_loader=train_loader,
               epochs=epochs,
               target_epsilon=target_epsilon,
               target_delta=target_delta,
               max_grad_norm=max_grad_norm
           )

           print(f"DP Training configured:")
           print(f"  Target epsilon: {target_epsilon}")
           print(f"  Target delta: {target_delta}")
           print(f"  Max grad norm: {max_grad_norm}")
           print(f"  Noise multiplier: {self.optimizer.noise_multiplier:.4f}")

       def train_epoch(self):
           self.model.train()
           total_loss = 0
           correct = 0
           total = 0

           for images, labels in self.train_loader:
               images, labels = images.to(self.device), labels.to(self.device)

               self.optimizer.zero_grad()
               outputs = self.model(images)
               loss = self.criterion(outputs, labels)
               loss.backward()
               self.optimizer.step()

               total_loss += loss.item() * images.size(0)
               _, predicted = outputs.max(1)
               total += labels.size(0)
               correct += predicted.eq(labels).sum().item()

           # Get current epsilon
           epsilon = self.privacy_engine.get_epsilon(delta=self.config['target_delta'])

           return total_loss / total, correct / total, epsilon

       def train(self, epochs):
           for epoch in range(epochs):
               train_loss, train_acc, epsilon = self.train_epoch()

               self.history['train_loss'].append(train_loss)
               self.history['train_acc'].append(train_acc)
               self.history['epsilon'].append(epsilon)

               print(f"Epoch {epoch+1}/{epochs}")
               print(f"  Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, ε: {epsilon:.2f}")

           return self.history

       def get_model_state(self):
           """Return model state dict (for non-private model)."""
           return self.model._module.state_dict() if hasattr(self.model, '_module') else self.model.state_dict()


   def train_with_dp(model, train_loader, val_loader, device, target_epsilon, epochs=10):
       """
       Convenience function to train with DP.
       """
       config = {
           'learning_rate': 1e-3,
           'target_delta': 1e-5,
           'max_grad_norm': 1.0
       }

       trainer = DPTrainer(model, device, config)
       trainer.attach_privacy_engine(
           train_loader,
           target_epsilon=target_epsilon,
           target_delta=config['target_delta'],
           epochs=epochs,
           max_grad_norm=config['max_grad_norm']
       )

       history = trainer.train(epochs)

       return trainer.get_model_state(), history
   ```

3. **Test DP training**
   ```python
   # Test script
   from model_side.models.cnn_model import SimpleCNN  # Use smaller model for DP
   from model_side.privacy.dp_training import train_with_dp

   model = SimpleCNN(num_classes=4)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Train with epsilon = 5
   model_state, history = train_with_dp(
       model, train_loader, val_loader, device,
       target_epsilon=5.0, epochs=10
   )

   torch.save(model_state, 'models/dp_epsilon_5.pth')
   ```

#### Deliverables:
- [ ] `src/privacy/dp_training.py`
- [ ] Working DP training pipeline

---

### Task 2.3: Run Privacy Experiments
**Duration:** Week 3-4
**Priority:** HIGH

#### Actionables:
1. **Train models with different privacy budgets**
   ```python
   epsilons = [1.0, 5.0, 10.0, float('inf')]  # inf = no DP

   for epsilon in epsilons:
       print(f"\n{'='*50}")
       print(f"Training with epsilon = {epsilon}")
       print('='*50)

       model = SimpleCNN(num_classes=4)

       if epsilon == float('inf'):
           # Standard training
           model_state = train_standard(model, train_loader, device, epochs=10)
       else:
           model_state, history = train_with_dp(
               model, train_loader, val_loader, device,
               target_epsilon=epsilon, epochs=10
           )

       torch.save(model_state, f'models/dp_epsilon_{epsilon}.pth')
   ```

2. **Evaluate each model**
   ```python
   results = []
   for epsilon in epsilons:
       model.load_state_dict(torch.load(f'models/dp_epsilon_{epsilon}.pth'))
       accuracy = evaluate(model, test_loader, device)
       results.append({'epsilon': epsilon, 'accuracy': accuracy})

   # Save results
   pd.DataFrame(results).to_csv('results/stage2/dp_privacy_utility.csv')
   ```

3. **Create privacy-utility trade-off curve**
   ```python
   import matplotlib.pyplot as plt

   df = pd.read_csv('results/stage2/dp_privacy_utility.csv')

   plt.figure(figsize=(10, 6))
   plt.plot(df['epsilon'], df['accuracy'], 'bo-', markersize=10)
   plt.xlabel('Privacy Budget (ε)', fontsize=12)
   plt.ylabel('Test Accuracy (%)', fontsize=12)
   plt.title('Privacy-Utility Trade-off')
   plt.xscale('log')
   plt.grid(True, alpha=0.3)
   plt.savefig('results/stage2/privacy_utility_tradeoff.png', dpi=150)
   ```

#### Deliverables:
- [ ] Models: `dp_epsilon_1.pth`, `dp_epsilon_5.pth`, `dp_epsilon_10.pth`
- [ ] `results/stage2/dp_privacy_utility.csv`
- [ ] `results/stage2/privacy_utility_tradeoff.png`

---

### Task 2.4: Integrate DP with Federated Learning
**Duration:** Week 4
**Priority:** HIGH

#### Actionables:
1. **Create DP-enhanced Flower client** `src/federated/client_dp.py`

2. **Implement local DP**
   ```python
   import flwr as fl
   import torch
   from opacus import PrivacyEngine
   from opacus.validators import ModuleValidator

   class COVIDxDPClient(fl.client.NumPyClient):
       """
       Flower client with local Differential Privacy.
       """
       def __init__(self, model, train_loader, val_loader, device, target_epsilon, target_delta=1e-5):
           # Make model DP-compatible
           self.model = ModuleValidator.fix(model).to(device)
           self.train_loader = train_loader
           self.val_loader = val_loader
           self.device = device
           self.target_epsilon = target_epsilon
           self.target_delta = target_delta

           self.criterion = torch.nn.CrossEntropyLoss()
           self.current_epsilon = 0

       def _setup_dp_training(self, local_epochs):
           """Set up DP training for this round."""
           optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

           privacy_engine = PrivacyEngine()
           model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
               module=self.model,
               optimizer=optimizer,
               data_loader=self.train_loader,
               epochs=local_epochs,
               target_epsilon=self.target_epsilon,
               target_delta=self.target_delta,
               max_grad_norm=1.0
           )

           return model, optimizer, train_loader, privacy_engine

       def fit(self, parameters, config):
           self.set_parameters(parameters)

           local_epochs = config.get("local_epochs", 1)

           # Set up DP for this round
           model, optimizer, train_loader, privacy_engine = self._setup_dp_training(local_epochs)

           model.train()
           total_loss = 0
           correct = 0
           total = 0

           for epoch in range(local_epochs):
               for images, labels in train_loader:
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

           # Update model reference
           self.model = model._module if hasattr(model, '_module') else model

           metrics = {
               "loss": total_loss / total,
               "accuracy": correct / total,
               "epsilon": self.current_epsilon
           }

           print(f"  DP Client - Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, ε: {self.current_epsilon:.2f}")

           return self.get_parameters(), total, metrics

       # ... rest of client methods
   ```

3. **Run DP + FL experiments**
   - Compare: Standard FL vs FL with Local DP
   - Test with different epsilon values

#### Deliverables:
- [ ] `src/federated/client_dp.py`
- [ ] DP + FL experiment results
- [ ] Comparison: FL vs FL+DP

---

### Task 2.5: Cross-Model Evaluation
**Duration:** Week 4-5
**Priority:** HIGH

#### Actionables:
1. **Share models with team**
   - Centralized baseline
   - Federated model
   - DP models (ε = 1, 5, 10)
   - DP + FL model

2. **Evaluate on teammates' data variants**
   - Use Member 1's data variants
   - Document results

3. **Analyze cross-evaluation results**
   - How does DP affect generalization?
   - Which epsilon provides best trade-off?

#### Deliverables:
- [ ] All model weights shared
- [ ] Cross-evaluation results
- [ ] Analysis of DP impact on different data

---

### Task 2.6: Stage 2 Report Contribution
**Duration:** Week 5
**Priority:** HIGH

#### Sections to Write (~6 pages):
1. **Differential Privacy Background** (1 page)
   - DP definition and guarantees
   - DP-SGD mechanism
   - Privacy accounting

2. **DP Implementation** (1.5 pages)
   - Opacus integration
   - Hyperparameter selection
   - Training considerations

3. **Privacy Experiments** (2 pages)
   - Experimental setup
   - Results for different epsilon values
   - Privacy-utility trade-off analysis
   - DP + FL results

4. **Cross-Evaluation Results** (1 page)
   - DP model performance on different data
   - Comparison with non-DP models

5. **Discussion** (0.5 pages)
   - Privacy guarantees achieved
   - Limitations
   - Recommendations

#### Deliverables:
- [ ] ~6 pages of final report content

---

## Communication Protocol

### With Member 1 (Data Lead):
- **Day 3:** Request sample data for model testing
- **Day 5:** Receive federated data splits
- **Week 4:** Provide models for cross-evaluation
- Share class weights needed for loss function

### With Member 3 (Evaluation Lead):
- **Week 2:** Provide centralized model for evaluation
- **Week 2:** Provide federated model for evaluation
- **Week 4:** Provide all DP models for final evaluation
- Coordinate on consistent evaluation methodology

### Models to Deliver:
| Model | When | Recipient |
|-------|------|-----------|
| Centralized baseline | Week 2, Day 5 | Member 3 |
| Federated model | Week 2, Day 7 | Member 3 |
| DP ε=1 model | Week 4 | Member 1, Member 3 |
| DP ε=5 model | Week 4 | Member 1, Member 3 |
| DP ε=10 model | Week 4 | Member 1, Member 3 |
| DP + FL model | Week 4 | Member 1, Member 3 |

---

## Time Estimates

| Task | Estimated Hours | Priority |
|------|-----------------|----------|
| Model architecture research | 3-4h | High |
| Base model implementation | 2-3h | High |
| Centralized training | 4-6h | High |
| Flower server | 3-4h | High |
| Flower client | 4-6h | High |
| FL experiments | 4-6h | High |
| Stage 1 report sections | 4-5h | High |
| DP fundamentals study | 2-3h | High |
| DP-SGD implementation | 6-8h | High |
| Privacy experiments | 6-8h | High |
| DP + FL integration | 4-6h | High |
| Cross-evaluation | 4-6h | High |
| Stage 2 report sections | 8-10h | High |
| **Total** | **~55-70h** | |

---

## Checklist

### Stage 1
- [ ] Research model architectures
- [ ] Implement CNN model
- [ ] Train centralized baseline
- [ ] Implement Flower server
- [ ] Implement Flower client
- [ ] Run FL experiments (3 clients)
- [ ] Compare centralized vs federated
- [ ] Write Stage 1 report sections

### Stage 2
- [ ] Study differential privacy fundamentals
- [ ] Implement DP-SGD with Opacus
- [ ] Train models with ε = 1, 5, 10
- [ ] Create privacy-utility trade-off curve
- [ ] Integrate DP with Flower client
- [ ] Run DP + FL experiments
- [ ] Cross-evaluate DP models
- [ ] Write Stage 2 report sections
