import os
import json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

class ClothingDataset(Dataset):
    def __init__(self, image_dir, label_mappings, transform=None):
        """
        Dataset for clothing images with JSON labels for multiple tasks.
        Args:
            image_dir (str): Directory containing images and JSONs.
            label_mappings (dict): Nested dictionary mapping clothing types and tasks to class labels.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_dir = image_dir
        self.label_mappings = label_mappings
        self.transform = transform
        self.data = []

        # Collect image paths, corresponding JSONs, and clothing types
        for clothing_type in os.listdir(image_dir):
            clothing_path = os.path.join(image_dir, clothing_type)
            if os.path.isdir(clothing_path):
                for fname in os.listdir(clothing_path):
                    if fname.endswith(".jpg"):  # Assuming .jpg for images
                        img_path = os.path.join(clothing_path, fname)
                        json_path = os.path.splitext(img_path)[0] + ".json"
                        if os.path.exists(json_path):
                            # Sample from a uniform distribution
                            if random.uniform(0, 1) > 0.7:
                                self.data.append((img_path, json_path, clothing_type))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, json_path, clothing_type = self.data[idx]

        # Load and transform the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Parse the JSON file for task labels
        with open(json_path, 'r') as f:
            item = json.load(f)

        try:
            # Dynamically map labels based on clothing type and mappings
            task_1 = item["Task 1"]
            task_2 = item["Task 2"]
            task_3 = item["Task 3"]

            # Check if all tasks are valid
            if None in (task_1, task_2, task_3):
                raise ValueError(
                    f"Invalid mapping in {json_path} for {clothing_type}: "
                    f"Task1: {task_1}, Task2: {task_2}, Task3: {task_3}"
                )

            task_labels = {"task1": task_1, "task2": task_2, "task3": task_3}

        except KeyError as e:
            raise ValueError(f"Missing task key in {json_path}: {e}")

        return image, task_labels, clothing_type


def get_dataloaders(data_dir, label_mappings, batch_size, num_workers=4):
    """
    Create DataLoaders for training, validation, and testing.
    Args:
        data_dir (str): Root directory containing 'train', 'val', 'test'.
        label_mappings (dict): Label mappings for all clothing types and tasks.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
    Returns:
        dict: DataLoaders for 'train', 'val', and 'test'.
    """
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Resize for EfficientNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = {
        split: ClothingDataset(os.path.join(data_dir, split), label_mappings, transform=transform)
        for split in ['train', 'val', 'test']
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True  # Enable pin_memory for faster GPU data transfer
        )
        for split in datasets
    }

    return dataloaders
