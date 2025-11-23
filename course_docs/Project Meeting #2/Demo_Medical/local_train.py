# import flwr as fl
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

# -------------------------------
# Argument Parser
# -------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description="Flower Template Client")
    parser.add_argument(
        "--client_id",
        type=int,
        required=True,
        help="The unique ID of the client",
    )
    return parser.parse_args()

# -------------------------------
# Dataset class
# -------------------------------
class PneumoMNISTDataset_Train(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        dataset = np.load(path)
        images, labels = dataset["train_images"], dataset["train_labels"]
        self.images = images
        self.labels = labels

        self.mean = np.mean(images)
        self.std = np.std(images)

        self.transform = transform

    def normalize(self, image):
        return (image - self.mean) / self.std
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = self.normalize(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)
    
    def __len__(self):
        return len(self.images)

class PneumoMNISTDataset_Test(torch.utils.data.Dataset):
    def __init__(self, path, mean, std, transform=None):
        dataset = np.load(path)
        images, labels = dataset["test_images"], dataset["test_labels"]
        self.images = images
        self.labels = labels

        self.mean = mean
        self.std = std
        self.transform = transform

    def normalize(self, image):
        return (image - self.mean) / self.std
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = self.normalize(image)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
    def __len__(self):
        return len(self.images)


# -------------------------------
# ResNet Architecture
# -------------------------------
class SimpleResNet(nn.Module):
    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2916, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
# -------------------------------
# Evaluation Loop
# -------------------------------
def evaluate(model, dataloader, device, criterion, split_name="Test"):
    model.eval()
    total_loss = 0.0
    correiterct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Evaluating... - {split_name} Loss: {avg_loss:.4f}, {split_name} Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# -------------------------------
# Train Loop
# -------------------------------
def train(model, train_dataloader, test_dataloader, device, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):  # Train for 2 epochs for demonstration
        epoch_loss = 0.0
        iterations = 0
        for batch_images, batch_labels in train_dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iterations += 1
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/iterations:.4f}")

        evaluate(model, train_dataloader, device, criterion, split_name="Train")
        evaluate(model, test_dataloader, device, criterion, split_name="Test")

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 32
NO_EPOCHS = 10

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Set all seeds to 42 for reproducibility
    fixed_seed = 42
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)

    args = get_arguments()
    data_path = f"client_{args.client_id}_data.npz"

    general_transformation = torch.nn.Sequential(
        transforms.Grayscale(num_output_channels=3),
    )

    transformation = torch.nn.Sequential(
        general_transformation,
        transforms.RandomRotation(degrees=15)
    )
         
    train_dataset = PneumoMNISTDataset_Train(data_path, transform=transformation)
    test_dataset = PneumoMNISTDataset_Test(data_path, train_dataset.mean, train_dataset.std, transform=general_transformation)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    image, label = next(iter(train_dataloader))

    criterion = nn.MSELoss()

    model = SimpleResNet()
    # print(model(image).shape)  # Should output torch.Size([32, 2])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train(model, train_dataloader, test_dataloader, device, criterion, optimizer, epochs=NO_EPOCHS)
