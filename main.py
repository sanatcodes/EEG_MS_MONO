import argparse
from pathlib import Path
import logging
import sys

from configs.base_config import BaseConfig
from configs.topo_config import TopoConfig
from configs.cae_config import CAEConfig
from configs.cluster_config import ClusterConfig
from utils.run_training import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='EEG Microstate Analysis Pipeline')
    parser.add_argument('--stage', type=str, required=True,
                       choices=['topo', 'cae', 'cluster', 'all'],
                       help='Pipeline stage to run')
    parser.add_argument('--config', type=Path, default=None,
                       help='Path to custom configuration file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load appropriate configuration
    if args.stage == 'topo' or args.stage == 'all':
        logger.info("Running topographic map generation...")
        # Import and run topomap generation
        
    if args.stage == 'cae' or args.stage == 'all':
        logger.info("Running CAE training...")
        # Call the train function with parameters
        train(
            config=None,  # Pass the appropriate config
            base_path=args.config,  # Use the config argument as base_path
            state='resting_state',  # Example state
            map_type='continuous',  # Example map_type
            val_split=0.2  # Example validation split
        )
        
    if args.stage == 'cluster' or args.stage == 'all':
        logger.info("Running clustering analysis...")
        # Import and run clustering
    
    logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    main() 