import os
import argparse
import logging
from pathlib import Path
import wandb
from datetime import datetime

from utils.create_topomaps import EEGProcessor
from utils.dataloader import TopomapDataset
from utils.models.cae import CAE
from utils.training import train_model
from utils.configs.data_config import DataConfig
from utils.configs.model_config import ModelConfig, TrainingConfig
from utils.configs.base_config import BaseConfig

def setup_logging(log_dir="logs"):
    """Configure logging with timestamp"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/run_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_topomaps(data_config, logger, is_local: bool):
    """Create topomaps from raw EEG data"""
    from utils.create_topomaps import main as create_topomaps_main
    
    logger.info("Creating topomaps from EEG data...")
    create_topomaps_main(
        local=is_local,
        process_mode=data_config.process_mode,
        data_dir=data_config.eeg_data_path,
        output_base=data_config.topomap_output_path
    )

def train(base_config: BaseConfig, model_config: ModelConfig, 
          train_config: TrainingConfig, logger):
    """Train the model with given configuration"""
    # Initialize wandb
    wandb.init(
        project=base_config.wandb_project,
        entity=base_config.wandb_entity,
        config={
            'model': vars(model_config),
            'training': vars(train_config)
        }
    )
    
    # Construct the correct path based on environment
    base_path = (train_config.topomap_path / 'local_run' 
                if train_config.is_local 
                else train_config.topomap_path)
    
    logger.info(f"Loading dataset from: {base_path}")
    
    # Create or load dataset
    dataset = TopomapDataset(
        base_path=base_path,
        state=train_config.state,
        map_type=train_config.map_type
    )
    
    # Initialize model
    model = CAE(model_config)
    
    # Train model
    train_model(
        model=model,
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        config=train_config
    )

def main():
    parser = argparse.ArgumentParser(description='EEG Topomap Processing and Model Training')
    parser.add_argument('--mode', choices=['create_data', 'train', 'both'], required=True,
                       help='Operation mode')
    parser.add_argument('--state', choices=['resting_state', 'task_state'],
                       help='State to process/train on')
    parser.add_argument('--map_type', choices=['continuous', 'peaks'],
                       help='Type of topomaps to use')
    parser.add_argument('--data_dir', type=Path, required=True,
                       help='Directory containing raw EEG data')
    parser.add_argument('--topomap_dir', type=Path,
                       help='Directory containing existing topomaps (for training) or where to save new ones')
    parser.add_argument('--model_dir', type=Path,
                       help='Directory where trained models will be saved')
    parser.add_argument('--local', action='store_true',
                       help='Flag to indicate local environment')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Modified base config
    topomap_dir = args.topomap_dir or (args.data_dir / 'topomaps')
    model_dir = args.model_dir or (args.data_dir / 'models')
    
    base_config = BaseConfig(
        data_dir=args.data_dir,
        output_dir=model_dir
    )
    
    try:
        if args.mode in ['create_data', 'both']:
            logger.info("Creating topomaps from EEG data...")
            data_config = DataConfig(
                eeg_data_path=base_config.data_dir,
                topomap_output_path=topomap_dir,
                state=args.state or 'resting_state',
                process_mode=args.map_type or 'both'
            )
            create_topomaps(data_config, logger, args.local)
            
        if args.mode in ['train', 'both']:
            logger.info("Starting model training...")
            model_config = ModelConfig()
            train_config = TrainingConfig(
                topomap_path=topomap_dir,
                state=args.state or 'resting_state',
                map_type=args.map_type or 'continuous',
                is_local=args.local
            )
            train(base_config, model_config, train_config, logger)
            
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        raise
    
    logger.info("Execution completed successfully")

if __name__ == "__main__":
    main() 