import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from tqdm import tqdm

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
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            momentum=config.get('momentum', 0.9)
        )

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

        for images, labels in tqdm(self.train_loader, desc="Training Epoch"):
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
        """Return model state dict (unwraps PrivacyEngine wrapper if needed)."""
        return self.model._module.state_dict() if hasattr(self.model, '_module') else self.model.state_dict()


def train_with_dp(model, train_loader, device, target_epsilon, target_delta=1e-5, epochs=10, max_grad_norm=1.0):
    """
    Convenience function to train with Differential Privacy.
    """
    config = {
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'target_delta': target_delta,
    }

    trainer = DPTrainer(model, device, config)
    trainer.attach_privacy_engine(
        train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm
    )

    history = trainer.train(epochs)

    return trainer.get_model_state(), history
