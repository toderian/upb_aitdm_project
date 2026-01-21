import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import io
import os

# --- CONFIGURATION ---
ZIP_FILE_PATH = "archive.zip"
IMG_SIZE = (224, 224)   
BATCH_SIZE = 32

class COVIDxZipDataset(Dataset):
    def __init__(self, zip_path, txt_file, source_filter=None, transform=None):
        self.zip_path = zip_path
        self.transform = transform
        self.txt_file = txt_file
        self.archive = None  # Cache for the zip file handle

        self.img_folder = txt_file.replace('.txt', '') + '/'

        # Read Metadata immediately (this is fast enough to do once)
        with zipfile.ZipFile(zip_path, 'r') as archive:
            with archive.open(txt_file) as f:
                self.df = pd.read_csv(f, sep=' ', header=None,
                                      names=['pid', 'filename', 'label', 'source'])

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
        # Lazy loading of the zip file. 
        # This ensures each worker process opens its own handle once and keeps it.
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.zip_path, 'r')

        row = self.df.iloc[idx]
        img_name = row['filename']
        label_str = row['label']

        try:
            path_in_zip = f"{self.img_folder}{img_name}"
            img_bytes = self.archive.read(path_in_zip)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_name}: {e}")
            # Return a black image on error to prevent crashing
            img = Image.new('RGB', IMG_SIZE)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.label_map.get(label_str, 0), dtype=torch.long)
        return img, label

    # Important: Close archive when dataset is destroyed (optional but good practice)
    def __del__(self):
        if self.archive:
            self.archive.close()

# --- HELPER 1: Standard Transforms ---
def get_standard_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- HELPER 2: Client Definitions ---
def _get_client_sources(client_id):
    client_mapping = {
        0: ['bimcv'],
        1: ['stonybrook', 'rsna'],
        2: ['sirm', 'ricord', 'cohen', 'actmed', 'fig1']
    }
    if client_id not in client_mapping:
        raise ValueError(f"Invalid Client ID. Options: {list(client_mapping.keys())}")
    return client_mapping[client_id]

# === PRIMARY FUNCTIONS ===

def get_federated_client(client_id, batch_size=BATCH_SIZE):
    transform = get_standard_transform()
    target_sources = _get_client_sources(client_id)

    dataset = COVIDxZipDataset(
        ZIP_FILE_PATH, "train.txt", source_filter=target_sources, transform=transform)

    targets = dataset.df['label'].map(dataset.label_map).values
    class_counts = torch.bincount(torch.tensor(targets))
    
    if len(class_counts) < 2:
        class_weights = torch.ones_like(torch.tensor(targets, dtype=torch.float))
    else:
        class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    print(f"[Client {client_id}] Train | Sources: {target_sources} | Count: {len(dataset)}")
    
    # Num workers is important here. 
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

def get_client_validation(client_id, batch_size=BATCH_SIZE):
    transform = get_standard_transform()
    target_sources = _get_client_sources(client_id)
    dataset = COVIDxZipDataset(ZIP_FILE_PATH, "val.txt", source_filter=target_sources, transform=transform)
    print(f"[Client {client_id}] Val | Sources: {target_sources} | Count: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def get_global_test_loader(batch_size=BATCH_SIZE):
    transform = get_standard_transform()
    dataset = COVIDxZipDataset(ZIP_FILE_PATH, "test.txt", source_filter=None, transform=transform)
    print(f"[Server] Global Test Set | Count: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# --- SELF-TEST ---
if __name__ == "__main__":
    if os.path.exists(ZIP_FILE_PATH):
        # Test Train
        try:
            dl = get_federated_client(2)
            print("Client 0 Train: OK")
        except Exception as e:
            print(f"Train Error: {e}")

        # Test Val
        try:
            dl = get_client_validation(2)
            print("Client 0 Val: OK")
        except Exception as e:
            print(f"Val Error: {e}")

        # Test Global
        try:
            dl = get_global_test_loader()
            print("Global Test: OK")
        except Exception as e:
            print(f"Global Test Error: {e}")
