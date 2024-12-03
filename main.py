import argparse
from pathlib import Path
import logging
import sys

from configs.base_config import BaseConfig
from configs.topo_config import TopoConfig
from configs.cae_config import CAEConfig
from configs.cluster_config import ClusterConfig

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
        # Import and run CAE training
        
    if args.stage == 'cluster' or args.stage == 'all':
        logger.info("Running clustering analysis...")
        # Import and run clustering
    
    logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    main() 