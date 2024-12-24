from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import os.path as osp
import cv2
import numpy as np
import os

class CustomImageDataset(Dataset):
    def __init__(self, annos, transform=None, train=True):
        """        
        annos: DataFrame([filename, folder_name_as_label])
        """
        self.annos = annos
        self.transform = transform
        self.train = train
        self.label_to_folder = {
            0: 'Classic', 1: 'Chic', 2: 'Cityboy', 
            3: 'Casual', 4: 'Minimal', 5: 'Workwear', 
            6: 'Retro', 7: 'Street', 8: 'Gorpcore', 
            9: 'Sporty', 10: 'Romantic', 11: 'Girlish', 12: 'Preppy'
        }


    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        target_row = self.annos.loc[idx]
        # Make image path
        filename = target_row['file']
        folder = self.label_to_folder[target_row['folder']]
        img_path = f'/root/25th-conference-EvenT/dataset/Musinsa_dataset/{folder}/{filename}'
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # image = Image.open(img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image file is corrupted or could not be read: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        
        return image, target_row['folder']