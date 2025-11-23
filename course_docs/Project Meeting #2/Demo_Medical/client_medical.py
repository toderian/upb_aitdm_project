import flwr as fl
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
from sklearn.metrics import accuracy_score


# -------------------------------
# Argument Parser
# -------------------------------

def get_arguments():
    parser = argparse.ArgumentParser(description="Flower Template Client")
    parser.add_argument(
        "--server_address",
        type=str,
        required=True,
        help="The server address",
    )

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
            # Import Resnet18
            # torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False),


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
# Hyperparameters
# -------------------------------
BATCH_SIZE = 32
NO_EPOCHS = 10

# -------------------------------
# Flower client class
# -------------------------------
class CustomFlowerClient(fl.client.NumPyClient):
    """
    Flower client implementation for federated learning.
    """
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """
        Train model on local client data and report loss + accuracy
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        epochs = int(config.get("local_epochs", 1))
        for _ in range(epochs):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).float()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()

        # Evaluate on local validation set
        loss_avg, _, metrics = self.evaluate(self.get_parameters(), config)
        print("\n\n\n==============================")
        print(f"Client evaluation - Loss: {loss_avg}, Accuracy: {metrics['accuracy']}")
        print("==============================\n\n")
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        """
        Evaluate model on local validation set
        """
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        ys, ypred = [], []
        loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).long()
                preds = self.model(X_batch)
                loss_sum += self.criterion(preds, y_batch).item() * X_batch.size(0)
                ys.extend(y_batch.cpu().numpy())
                ypred.extend(preds.argmax(dim=1).cpu().numpy())
        if len(ys) == 0:
            return float("inf"), 0, {"accuracy": 0.0}
        acc = accuracy_score(ys, ypred)
        return float(loss_sum / len(ys)), len(ys), {"accuracy": acc}



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

    model = SimpleResNet()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=CustomFlowerClient(model, train_dataloader, test_dataloader, DEVICE).to_client()
    )
