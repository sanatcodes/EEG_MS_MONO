import os
import wandb
import torch
import numpy as np
import random
import logging
from torch.utils.data import DataLoader
from dataloader import TopomapDataset
from models.cae import CAE 
from training import train_model
from configs.training_sweep_config import SWEEP_CONFIG
from pathlib import Path
import argparse

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Set random seed to {seed}")

def create_datasets(base_path, state='resting_state', map_type='continuous', val_split=0.2):
    """
    Create train and validation datasets
    
    Args:
        base_path (str): Path to topomaps directory
        state (str): 'resting_state', 'task_state', or 'both'
        map_type (str): 'continuous' or 'peaks'
        val_split (float): Fraction of data to use for validation
    """
    logger.info(f"Creating datasets with base_path={base_path}, state={state}, map_type={map_type}")
    
    datasets = []
    states = ['resting_state', 'task_state'] if state == 'both' else [state]
    logger.debug(f"Processing states: {states}")
    
    for current_state in states:
        logger.debug(f"Creating dataset for state: {current_state}")
        try:
            dataset = TopomapDataset(base_path, state=current_state, map_type=map_type)
            logger.info(f"Created dataset for {current_state} with {len(dataset)} samples")
            datasets.append(dataset)
        except Exception as e:
            logger.error(f"Failed to create dataset for {current_state}: {str(e)}")
            raise
    
    # Combine datasets if using both states
    if len(datasets) > 1:
        # Implement dataset combination logic if needed
        logger.warning("Multiple dataset combination not yet implemented")
        pass
    else:
        combined_dataset = datasets[0]
        logger.debug(f"Using single dataset with {len(combined_dataset)} samples")
    
    # Calculate split indices
    dataset_size = len(combined_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    logger.info(f"Splitting dataset - Total: {dataset_size}, Train: {train_size}, Val: {val_size}")
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.debug("Dataset split complete")
    logger.info(f"Final dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def train(config=None, base_path=None, state='resting_state', map_type='continuous', val_split=0.2):
    try:
        with wandb.init(config=config) as run:
            config = wandb.config
            set_seed(42)
            
            # Log device information
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Use provided base path or default
            if not base_path:
                base_path = '/home/CAMPUS/d18129674/EEG_DC_TOPO/TOPOMAPS_OUT_3'
                logger.warning(f"TOPOMAP_PATH not set, using default: {base_path}")
            
            logger.info(f"Using data path: {base_path}")
            if not os.path.exists(base_path):
                raise ValueError(f"Data path does not exist: {base_path}")
            
            # Create datasets with error handling
            try:
                train_dataset, val_dataset = create_datasets(base_path, state=state, map_type=map_type, val_split=val_split)
                logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            except Exception as e:
                logger.error(f"Failed to create datasets: {str(e)}")
                raise
            
            # Validate datasets are not empty
            if len(train_dataset) == 0 or len(val_dataset) == 0:
                raise ValueError("One or both datasets are empty")
            
            # Create dataloaders with memory pinning only on CUDA
            use_pin_memory = torch.cuda.is_available()
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=use_pin_memory
            )
            
            # Initialize model and move to device
            model = CAE(config)
            model = model.to(device)
            
            # Log model summary
            logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            best_val_loss = train_model(model, train_loader, val_loader, config)
            
            # Log final metrics
            wandb.log({
                "best_val_loss": best_val_loss,
                "final_learning_rate": config.learning_rate,
                "train_dataset_size": len(train_dataset),
                "val_dataset_size": len(val_dataset),
                "total_parameters": sum(p.numel() for p in model.parameters())
            })
            
            return best_val_loss
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        # Log the error to wandb
        if wandb.run is not None:
            wandb.run.summary["error"] = str(e)
        raise

def main():
    try:
        # Add argument parsing
        parser = argparse.ArgumentParser(description='Train CAE model')
        parser.add_argument('--base_path', type=str, default=None,
                          help='Path to topomaps directory')
        parser.add_argument('--state', type=str, default='resting_state',
                          choices=['resting_state', 'task_state', 'both'],
                          help='State to use for training')
        parser.add_argument('--map_type', type=str, default='continuous',
                          choices=['continuous', 'peaks'],
                          help='Type of topomap to use')
        parser.add_argument('--val_split', type=float, default=0.2,
                          help='Validation split ratio')
        parser.add_argument('--sweep_count', type=int, default=5,
                          help='Number of sweep runs')
        args = parser.parse_args()

        # Ensure wandb is logged in
        if not wandb.api.api_key:
            logger.error("WandB API key not found. Please log in or set WANDB_API_KEY.")
            return
            
        sweep_id = wandb.sweep(
            sweep=SWEEP_CONFIG,
            project="Simplified-Microstate-CAE"
        )
        
        logger.info(f"Created sweep with ID: {sweep_id}")
        
        # Save sweep ID to file for reference
        with open("sweep_id.txt", "w") as f:
            f.write(sweep_id)
            
        # Modified train function to use command line arguments
        train_func = lambda: train(
            config=None,
            base_path=args.base_path,
            state=args.state,
            map_type=args.map_type,
            val_split=args.val_split
        )
            
        wandb.agent(sweep_id, function=train_func, count=args.sweep_count)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()