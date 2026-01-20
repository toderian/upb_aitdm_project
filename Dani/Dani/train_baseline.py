import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm  # Import for loading bar
from data_loader import get_federated_client
import sys

# --- CONFIGURATION ---
CLIENT_ID_TO_TRAIN = 0  # We train on "The Giant" (Client 0) as our baseline
EPOCHS = 3              # Keep it short for a quick test
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_stats():
    """Helper to get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        # memory_allocated returns bytes, convert to GB
        mem = torch.cuda.memory_allocated(DEVICE) / 1024**3
        return f"{mem:.2f}GB"
    return "N/A"


def train_baseline():
    print(
        f"--- Member 1: Starting Baseline Training on Client {CLIENT_ID_TO_TRAIN} ---")
    print(f"Using Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    # 1. Get the Data (Using your Federated Split)
    train_loader = get_federated_client(
        client_id=CLIENT_ID_TO_TRAIN, batch_size=64)

    # 2. Define the Model (ResNet18)
    model = models.resnet18(pretrained=True)

    # Modify the final layer for Binary Classification (Positive vs Negative)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(DEVICE)

    # 3. Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    model.train()

    for epoch in range(EPOCHS):
        # Reset peak stats for this epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        running_loss = 0.0
        correct = 0
        total = 0

        # Create a progress bar wrapper around the data loader
        loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar with current stats + GPU Usage
            current_acc = 100 * correct / total
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{current_acc:.2f}%",
                gpu=get_gpu_stats()
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Get Peak GPU usage for this epoch
        peak_mem = "N/A"
        if torch.cuda.is_available():
            peak_mem = f"{torch.cuda.max_memory_allocated(DEVICE) / 1024**3:.2f}GB"

        # Print summary at the end of the epoch
        print(
            f"Epoch {epoch+1} Summary -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | Peak GPU: {peak_mem}")

    # 5. Save the Baseline Weights
    save_path = "baseline_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining Complete. Weights saved to: {save_path}")
    print("-> Hand this file to Member 3 for evaluation setup.")


if __name__ == "__main__":
    train_baseline()
