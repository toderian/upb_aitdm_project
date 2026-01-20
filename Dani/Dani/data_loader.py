import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os

# --- CONFIGURATION (Member 1 Settings) ---
# Ensure these match your actual file names
ZIP_FILE_PATH = "archive.zip"
TXT_FILE_NAME = "train.txt"
IMG_SIZE = (224, 224)   # Standard for ResNet/VGG
BATCH_SIZE = 32


class COVIDxZipDataset(Dataset):
    """
    Custom Dataset created by Member 1.
    Reads images directly from ZIP to save disk space.
    Filters data based on 'source' for Federated Non-IID splitting.
    """

    def __init__(self, zip_path, txt_file, source_filter=None, transform=None):
        self.zip_path = zip_path
        self.transform = transform

        # 1. Read Metadata from Text File inside Zip
        # We open the zip just once here to read the CSV into memory
        with zipfile.ZipFile(zip_path, 'r') as archive:
            with archive.open(txt_file) as f:
                # Format: [patient_id] [filename] [label] [data_source]
                self.df = pd.read_csv(f, sep=' ', header=None,
                                      names=['pid', 'filename', 'label', 'source'])

        # 2. Filter Logic (The Non-IID Split)
        # If a specific list of sources is requested, keep only those rows
        if source_filter:
            if isinstance(source_filter, list):
                self.df = self.df[self.df['source'].isin(source_filter)]
            else:
                self.df = self.df[self.df['source'] == source_filter]

            # Reset index so __getitem__ works correctly
            self.df = self.df.reset_index(drop=True)

        # 3. Label Mapping (Binary: Positive vs Negative)
        self.label_map = {'positive': 1, 'negative': 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row data
        row = self.df.iloc[idx]
        img_name = row['filename']
        label_str = row['label']

        # 4. Open Zip & Extract Image Bytes
        # We open the zip file *inside* getitem to ensure thread safety with multiple workers
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            try:
                # Try reading assuming it might be in a 'train' subfolder
                img_bytes = archive.read(f"train/{img_name}")
            except KeyError:
                try:
                    # Try reading from root
                    img_bytes = archive.read(img_name)
                except KeyError:
                    # Last resort fallback (sometimes filenames have paths)
                    print(f"Error finding {img_name} in zip.")
                    # Return a black image to prevent crash, but log error
                    return torch.zeros((3, 224, 224)), torch.tensor(0)

        # Convert bytes to PIL Image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Apply Transforms
        if self.transform:
            img = self.transform(img)

        # Convert Label
        label = torch.tensor(self.label_map.get(
            label_str, 0), dtype=torch.long)

        return img, label


def get_federated_client(client_id, batch_size=BATCH_SIZE):
    """
    Helper function for Team Members.
    Returns a DataLoader for a specific 'Virtual Client'.
    """
    # Standard Preprocessing
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # --- MEMBER 1 SPLIT STRATEGY (Non-IID) ---
    # Based on the distribution analysis:
    # Client 0 (The Giant): BIMCV (~43k images)
    # Client 1 (The Partners): Stonybrook + RSNA (~22k images)
    # Client 2 (The Edge): Small clinics (SIRM, RICORD, Cohen, ActMed, Fig1) (~2.3k images)

    client_mapping = {
        0: ['bimcv'],
        1: ['stonybrook', 'rsna'],
        2: ['sirm', 'ricord', 'cohen', 'actmed', 'fig1']
    }

    if client_id not in client_mapping:
        raise ValueError(
            f"Invalid Client ID. Options: {list(client_mapping.keys())}")

    target_sources = client_mapping[client_id]

    # Initialize Dataset
    dataset = COVIDxZipDataset(
        zip_path=ZIP_FILE_PATH,
        txt_file=TXT_FILE_NAME,
        source_filter=target_sources,
        transform=transform
    )

    print(
        f"[Client {client_id}] Loading Sources: {target_sources} | Count: {len(dataset)}")

    # UPDATED: Added num_workers and pin_memory for speed
    # num_workers=0 means main process. 2-4 is usually good for Windows.
    # pin_memory=True speeds up CPU->GPU transfer.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # Parallel loading
        pin_memory=True     # Faster GPU transfer
    )


# --- SANITY CHECK (Run this file directly to test) ---
if __name__ == "__main__":
    print("--- Member 1: Experiment Design & Data Check ---")

    # Check if zip exists
    if not os.path.exists(ZIP_FILE_PATH):
        print(
            f"ERROR: {ZIP_FILE_PATH} not found. Please put the zip file in this directory.")
    else:
        # 1. Test Client 0 (The Giant)
        try:
            loader_0 = get_federated_client(0)
            data, label = next(iter(loader_0))
            print(f"Client 0 Batch Shape: {data.shape}")
            print("Client 0 successfully loaded!\n")
        except Exception as e:
            print(f"Client 0 Failed: {e}")

        # 2. Test Client 2 (The Small/Edge Client)
        try:
            loader_2 = get_federated_client(2)
            data, label = next(iter(loader_2))
            print(f"Client 2 Batch Shape: {data.shape}")
            print("Client 2 successfully loaded!\n")
        except Exception as e:
            print(f"Client 2 Failed: {e}")

        print("--- Data Loader Verification Complete ---")
