import zipfile
import pandas as pd
from PIL import Image
import io
import torch
from torch.utils.data import Dataset


class COVIDxZipDataset(Dataset):
    def __init__(self, zip_path, txt_file_name, transform=None, source_filter=None):
        """
        Args:
            zip_path: Path to the .zip file (e.g., 'data.zip')
            txt_file_name: Name of the text file inside zip (e.g., 'train.txt')
            transform: Image transformations (resize, normalize)
            source_filter: If set (e.g., 'rsna'), only loads that client's data.
        """
        self.zip_path = zip_path
        self.transform = transform

        # 1. Open Zip temporarily just to read the text file
        with zipfile.ZipFile(zip_path, 'r') as archive:
            with archive.open(txt_file_name) as f:
                # Read the metadata into a DataFrame
                self.df = pd.read_csv(f, sep=' ', header=None,
                                      names=['pid', 'filename', 'label', 'source'])

        # 2. Filter for Federated Split (Member 1 Responsibility)
        if source_filter:
            self.df = self.df[self.df['source'] ==
                              source_filter].reset_index(drop=True)

        # Map string labels to integers (e.g., positive -> 1, negative -> 0)
        self.label_map = {'negative': 0, 'positive': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the filename from the dataframe
        img_name = self.df.iloc[idx]['filename']
        label_str = self.df.iloc[idx]['label']

        # 3. OPEN ZIP, EXTRACT BYTES, CONVERT TO IMAGE
        # We open the zip here to ensure thread safety if using multiple workers
        with zipfile.ZipFile(self.zip_path, 'r') as archive:
            # Note: You might need to adjust the path if images are in a subfolder inside zip
            # e.g., archive.read(f"train/{img_name}")
            img_bytes = archive.read(img_name)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.label_map.get(
            label_str, 0), dtype=torch.long)

        return image, label
