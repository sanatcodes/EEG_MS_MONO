from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Model architecture
    latent_dim: int = 32
    hidden_dims: List[int] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 64, 128]

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Required arguments should come first
    topomap_path: Path
    state: str
    map_type: str
    # Add local environment flag
    is_local: bool = False
    
    # Arguments with default values should come after
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    optimizer: Literal['adam', 'adamw'] = 'adam'
    weight_decay: float = 0.0001
    
    # Early stopping and scheduler
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    scheduler_factor: float = 0.2
    
    # Data settings
    validation_split: float = 0.2
    
    # Paths
    checkpoint_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Ensure paths are Path objects and create directories."""
        self.topomap_path = Path(self.topomap_path)
        
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.topomap_path.parent / 'checkpoints'
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True) 