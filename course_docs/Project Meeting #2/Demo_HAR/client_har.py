import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import flwr as fl

# -------------------------------
# Feature extractor for skeleton sequences
# -------------------------------
class SequenceFeatureExtractor:
    """
    Extract statistical features from skeleton sequences:
    mean, standard deviation, min, max.
    """
    def transform(self, frames: np.ndarray) -> np.ndarray:
        mean = frames.mean(axis=0).flatten()
        std = frames.std(axis=0).flatten()
        min_ = frames.min(axis=0).flatten()
        max_ = frames.max(axis=0).flatten()
        return np.concatenate([mean, std, min_, max_]).astype(np.float32)

# -------------------------------
# Dataset class
# -------------------------------

# Lets say we have 1200 samples:

# client 1 - gets 0-399
# client 2 - gets 400-799
# client 3 - gets 800-1199

class SkeletonSequenceDataset:
    """
    Loads CSV data and converts each sequence into feature vectors.
    """
    def __init__(self, csv_path, extractor):
        df = pd.read_csv(csv_path)
        joint_cols = [c for c in df.columns if c.startswith('J')]
        groups = df.groupby('IDSample')
        X_list, y_list = [], []
        for _, g in groups:
            frames = g[joint_cols].values.astype(np.float32)
            frames3 = frames.reshape(frames.shape[0], 25, 3)
            feat = extractor.transform(frames3)
            X_list.append(feat)
            y_list.append(g['Action'].iloc[0])
        self.X = np.stack(X_list)
        self.y = np.array(y_list)
        # Encode labels if necessary
        if self.y.dtype == object or not np.issubdtype(self.y.dtype, np.number):
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)
            self.label_encoder = le
        else:
            self.label_encoder = None

# -------------------------------
# MLP classifier
# -------------------------------
class MLPClassifier(nn.Module):
    """
    Simple feed-forward network with 2 hidden layers and dropout.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

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
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).long()
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.optimizer.step()
        # Evaluate on local validation set
        loss_avg, _, metrics = self.evaluate(self.get_parameters(), config)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SequenceFeatureExtractor()
    dataset = SkeletonSequenceDataset(args.csv, extractor)

    tensor_X = torch.from_numpy(dataset.X)
    tensor_y = torch.from_numpy(dataset.y)

    # Split train/validation (80/20)
    n = len(dataset.X)
    n_val = max(1, int(0.2 * n))
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_loader = DataLoader(TensorDataset(tensor_X[train_idx], tensor_y[train_idx]), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(tensor_X[val_idx], tensor_y[val_idx]), batch_size=16, shuffle=False)

    model = MLPClassifier(in_dim=dataset.X.shape[1], num_classes=int(dataset.y.max()+1))

    # Start Flower client
    fl.client.start_client(
        server_address=args.server,
        client=CustomFlowerClient(model, train_loader, val_loader, DEVICE).to_client()
    )
