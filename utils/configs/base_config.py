from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class BaseConfig:
    """Base configuration class with common settings."""
    
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    
    # Logging settings
    wandb_project: str = "eeg-microstates"
    wandb_entity: Optional[str] = None
    log_interval: int = 100
    
    # Device settings
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True) 