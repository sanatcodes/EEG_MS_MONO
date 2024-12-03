import os
import wandb
import torch
import numpy as np
import random
import logging
from torch.utils.data import DataLoader
from src.data.dataset import create_datasets
from src.models.cae import CAE 
from src.utils.training import train_model
from config.sweep_config import SWEEP_CONFIG

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

def train(config=None):
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
            
            # Get and validate base path
            base_path = os.environ.get('TOPOMAP_PATH')
            if not base_path:
                base_path = '/home/CAMPUS/d18129674/EEG_DC_TOPO/TOPOMAPS_OUT_3'
                logger.warning(f"TOPOMAP_PATH not set, using default: {base_path}")
            
            logger.info(f"Using data path: {base_path}")
            if not os.path.exists(base_path):
                raise ValueError(f"Data path does not exist: {base_path}")
            
            # Create datasets with error handling
            try:
                csv_path = '~/EEG_DC_TOPO/EEG_data/subject-info.csv'
                train_dataset, val_dataset = create_datasets(base_path, csv_path=csv_path)
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
            
        wandb.agent(sweep_id, function=train, count=5)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()