import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class TopomapDataset(Dataset):
    def __init__(self, base_path, state='resting_state', map_type='continuous', 
                 train_split=0.8, batch_size=32):
        """
        Args:
            base_path (str): Path to the topomaps directory
            state (str): 'resting_state' or 'task_state'
            map_type (str): 'continuous' or 'peaks'
            train_split (float): Proportion of data to use for training
            batch_size (int): Batch size for data loaders
        """
        super().__init__()
        self.base_path = Path(base_path)
        self.state = state
        self.map_type = map_type
        
        # Validate inputs before building dataset
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
            
        if state not in ['resting_state', 'task_state']:
            raise ValueError(f"Invalid state: {state}. Must be 'resting_state' or 'task_state'")
            
        if map_type not in ['continuous', 'peaks']:
            raise ValueError(f"Invalid map_type: {map_type}. Must be 'continuous' or 'peaks'")
        
        # Build dataset structure
        self.samples = self._build_dataset()
        
        # Ensure we found some data
        if not self.samples:
            raise ValueError(f"No valid samples found in {self.base_path / self.map_type / self.state}")
        
        # Create train/val split
        total_size = len(self.samples)
        train_size = int(total_size * train_split)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            self, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Adjust based on your system
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Adjust based on your system
            pin_memory=True
        )
        
    def _build_dataset(self):
        """Build dataset structure from directory"""
        samples = []
        state_path = self.base_path / self.map_type / self.state
        
        if not state_path.exists():
            raise ValueError(f"Data directory not found: {state_path}")
        
        # Iterate through subject directories
        for subject_dir in state_path.glob("Subject*"):
            # Load metadata
            metadata_file = 'continuous_metadata.json' if self.map_type == 'continuous' else 'subject_metadata.json'
            metadata_path = subject_dir / metadata_file
            
            if not metadata_path.exists():
                continue
                
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Get subject ID from directory name
                subject_id = subject_dir.name
                
                # Add each topomap with its metadata
                timepoints = metadata.get('timepoints' if self.map_type == 'continuous' else 'peaks', [])
                
                for point in timepoints:
                    img_name = f'topo_continuous_{point["timepoint_id"]:04d}.png' if self.map_type == 'continuous' \
                             else f'topo_peak_{point["peak_id"]:04d}.png'
                    img_path = subject_dir / img_name
                    
                    if not img_path.exists():
                        continue
                        
                    samples.append({
                        'image_path': img_path,
                        'subject_id': subject_id,
                        'time_ms': point['time_ms'],
                        'index': point['timepoint_id'] if self.map_type == 'continuous' else point['peak_id']
                    })
                    
            except Exception as e:
                print(f"Error processing {subject_dir}: {e}")
                continue
                
        return samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict: Contains:
                - image: torch.Tensor of shape [1, H, W] normalized to [-1, 1]
                - subject_id: str
                - time_ms: float
                - index: int
        """
        sample = self.samples[idx]
        
        # Load and convert image to grayscale (topomaps are essentially grayscale)
        image = Image.open(sample['image_path']).convert('L')
        
        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        return {
            'image': image,
            'subject_id': sample['subject_id'],
            'time_ms': sample['time_ms'],
            'index': sample['index']
        }

    def get_stats(self):
        """Get dataset statistics"""
        n_subjects = len(set(s['subject_id'] for s in self.samples))
        n_images = len(self.samples)
        
        # Load first image to get dimensions
        first_image = Image.open(self.samples[0]['image_path'])
        image_size = first_image.size
        
        return {
            "num_subjects": n_subjects,
            "num_images": n_images,
            "image_size": image_size,
            "state": self.state,
            "map_type": self.map_type
        }
