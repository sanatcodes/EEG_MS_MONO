import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TopomapDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_names[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

    def get_stats(self):
        shapes = np.array([Image.open(os.path.join(self.folder_path, f)).size 
                          for f in self.file_names])
        return {
            "num_images": len(self.file_names),
            "mean_shape": shapes.mean(axis=0),
            "std_shape": shapes.std(axis=0)
        }
