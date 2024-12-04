from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

@dataclass
class DataConfig:
    """Configuration for data processing and topomap generation."""
    
    # Data paths
    eeg_data_path: Path
    topomap_output_path: Path
    
    # Processing settings
    process_mode: Literal['continuous', 'peaks', 'both'] = 'continuous'
    map_interval_ms: int = 40
    use_gfp_peaks: bool = True
    
    # GFP peak detection parameters
    gfp_threshold: float = 0.5
    min_peak_distance_ms: float = 40.0
    
    # Data states
    state: Literal['resting_state', 'task_state'] = 'resting_state'
    
    def __post_init__(self):
        """Ensure directories exist and paths are Path objects."""
        self.eeg_data_path = Path(self.eeg_data_path)
        self.topomap_output_path = Path(self.topomap_output_path)
        
        # Verify that eeg_data_path is a directory
        if not self.eeg_data_path.is_dir():
            raise ValueError(f"EEG data path must be a directory: {self.eeg_data_path}")
        
        # Create output directories
        self.topomap_output_path.mkdir(parents=True, exist_ok=True)
        (self.topomap_output_path / 'continuous').mkdir(exist_ok=True)
        (self.topomap_output_path / 'peaks').mkdir(exist_ok=True)
        
    def get_eeg_files(self):
        """Return a list of EEG files in the data directory."""
        # Adjust the pattern based on your EEG file extension (e.g., .edf, .set, etc.)
        return list(self.eeg_data_path.glob("*.edf"))
        