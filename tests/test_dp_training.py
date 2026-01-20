import torch
from torch.utils.data import TensorDataset, DataLoader
from model_side.models.cnn_model import SimpleCNN
from model_side.privacy.dp_training import train_with_dp


def test_dp_training():
    # Create dummy data (batch=8, 3 channels, 224x224)
    X = torch.randn(8, 3, 224, 224)
    y = torch.randint(0, 4, (8,))
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=4)

    # Initialize model
    model = SimpleCNN(num_classes=4)
    device = torch.device('cpu')

    # Train with DP
    print("Testing DP training with epsilon=5.0...")
    model_state, history = train_with_dp(
        model, train_loader, device,
        target_epsilon=5.0, target_delta=1e-5,
        epochs=1, max_grad_norm=1.0
    )

    print(f"Training complete. Final epsilon: {history['epsilon'][-1]:.2f}")
    print(f"Final accuracy: {history['train_acc'][-1]:.4f}")


if __name__ == '__main__':
    test_dp_training()