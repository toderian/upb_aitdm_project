import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import io
import os

# --- CONFIGURATION (Member 1 Settings) ---
# Ensure these match your actual file names
ZIP_FILE_PATH = "../../Dani/Dani/archive.zip"
IMG_SIZE = (224, 224)   # Standard for ResNet/VGG
BATCH_SIZE = 32


class COVIDxZipDataset(Dataset):
    def __init__(self, zip_path, txt_file, source_filter=None, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.txt_file = txt_file

        # 1. Infer the image folder based on the text file name
        # If txt_file is "train.txt", look in "train/" folder, etc.
        self.img_folder = txt_file.replace('.txt', '') + '/'

        # 2. Read Metadata
        with zipfile.ZipFile(zip_path, 'r') as archive:
            with archive.open(txt_file) as f:
                self.df = pd.read_csv(f, sep=' ', header=None,
                                      names=['pid', 'filename', 'label', 'source'])

        # 3. Filter Logic (Federated Split)
        if source_filter:
            if isinstance(source_filter, list):
                self.df = self.df[self.df['source'].isin(source_filter)]
            else:
                self.df = self.df[self.df['source'] == source_filter]
            self.df = self.df.reset_index(drop=True)

        self.label_map = {'positive': 1, 'negative': 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        label_str = row['label']

        # 4. Open Zip & Extract Image
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            try:
                # Use the dynamic folder path (e.g., "test/image.png")
                path_in_zip = f"{self.img_folder}{img_name}"
                img_bytes = archive.read(path_in_zip)
            except KeyError:
                print(f"Error: {path_in_zip} not found in zip.")
                return torch.zeros((3, 224, 224)), torch.tensor(0)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.label_map.get(
            label_str, 0), dtype=torch.long)
        return img, label

# --- HELPER 1: Standard Transforms ---


def get_standard_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

# --- HELPER 2: Client Definitions ---


def _get_client_sources(client_id):
    client_mapping = {
        0: ['bimcv'],
        1: ['stonybrook', 'rsna'],
        2: ['sirm', 'ricord', 'cohen', 'actmed', 'fig1']
    }
    if client_id not in client_mapping:
        raise ValueError(
            f"Invalid Client ID. Options: {list(client_mapping.keys())}")
    return client_mapping[client_id]

# === PRIMARY FUNCTIONS ===


def get_federated_client(client_id, batch_size=BATCH_SIZE):
    """
    TRAINING LOADER: Loads from 'train.txt' with Class Balancing (WeightedSampler)
    """
    transform = get_standard_transform()
    target_sources = _get_client_sources(client_id)

    dataset = COVIDxZipDataset(
        ZIP_FILE_PATH, "train.txt", source_filter=target_sources, transform=transform)

    # Calculate Weights for Imbalance Fix
    targets = dataset.df['label'].map(dataset.label_map).values
    class_counts = torch.bincount(torch.tensor(targets))
    # Avoid division by zero if a client has 0 negatives
    if len(class_counts) < 2:
        class_weights = torch.ones_like(
            torch.tensor(targets, dtype=torch.float))
    else:
        class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    print(
        f"[Client {client_id}] Training Set | Sources: {target_sources} | Count: {len(dataset)}")

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, pin_memory=True)


def get_client_validation(client_id, batch_size=BATCH_SIZE):
    """
    VALIDATION LOADER: Loads from 'val.txt', specific to the client's sources.
    No shuffling, no sampling. Used to check if local training is working.
    """
    transform = get_standard_transform()
    target_sources = _get_client_sources(client_id)

    # Note: Using 'val.txt'
    dataset = COVIDxZipDataset(
        ZIP_FILE_PATH, "val.txt", source_filter=target_sources, transform=transform)

    print(
        f"[Client {client_id}] Validation Set | Sources: {target_sources} | Count: {len(dataset)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def get_global_test_loader(batch_size=BATCH_SIZE):
    """
    GLOBAL TEST LOADER: Loads from 'test.txt'.
    Uses ALL sources. This is the 'Gold Standard' for the Server to evaluate.
    """
    transform = get_standard_transform()

    # Note: source_filter=None means "Use All Sources"
    dataset = COVIDxZipDataset(
        ZIP_FILE_PATH, "test.txt", source_filter=None, transform=transform)

    print(f"[Server] Global Test Set Loaded | Count: {len(dataset)}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


# --- SELF-TEST ---
if __name__ == "__main__":
    if os.path.exists(ZIP_FILE_PATH):
        # Test Train
        try:
            dl = get_federated_client(0)
            print("Client 0 Train: OK")
        except Exception as e:
            print(f"Train Error: {e}")

        # Test Val
        try:
            dl = get_client_validation(0)
            print("Client 0 Val: OK")
        except Exception as e:
            print(f"Val Error: {e}")

        # Test Global
        try:
            dl = get_global_test_loader()
            print("Global Test: OK")
        except Exception as e:
            print(f"Global Test Error: {e}")
