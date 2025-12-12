from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class Kadid10kDataset(Dataset):
    """
    Dataset dla FR-IQA używający KADID-10k.
    Każdy element zwraca:
    - reference_image
    - distorted_image
    - dmos_value
    """

    def __init__(self, images_folder_path, csv_file_path, transform=None):
        self.images_folder_path = Path(images_folder_path)
        self.transform = transform

        # Ładowanie tabeli DMOS
        self.dmos_table = pd.read_csv(csv_file_path)


    def __len__(self):
        return len(self.dmos_table)


    def __getitem__(self, index):
        row = self.dmos_table.iloc[index]

        reference_image_name = row['ref_img']
        distorted_image_name = row['dist_img']
        dmos_value = row['dmos']

        reference_image_path = self.images_folder_path / reference_image_name
        distorted_image_path = self.images_folder_path / distorted_image_name

        reference_image = Image.open(reference_image_path).convert('RGB')
        distorted_image = Image.open(distorted_image_path).convert('RGB')

        if self.transform:
            reference_image = self.transform(reference_image)
            distorted_image = self.transform(distorted_image)

        return reference_image, distorted_image, torch.tensor(dmos_value, dtype=torch.float32)
