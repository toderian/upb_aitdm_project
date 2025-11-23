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