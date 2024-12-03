import numpy as np
import logging
import mne
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Dict, Union, Optional

class GFPCalculator:
    def __init__(self, sampling_rate: float = 500.0):
        """Initialize GFP Calculator with configurable sampling rate."""
        self.sampling_rate = sampling_rate
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging with detailed formatting"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def calculate_gfp(self, data: np.ndarray) -> np.ndarray:
        """Calculate Global Field Power from EEG data."""
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data array, got shape {data.shape}")
        
        # Calculate mean across channels
        v_mean = np.mean(data, axis=0)
        
        # Calculate GFP using the standard formula
        gfp = np.sqrt(np.mean((data - v_mean) ** 2, axis=0))
        
        return gfp

    def find_gfp_peaks_and_windows(
        self, 
        data: np.ndarray,
        gfp_values: Optional[np.ndarray] = None,
        min_duration_ms: float = 40.0,
        window_size_ms: float = 40.0,
        gfp_threshold_pct: float = 0.3,
        prominence_pct: float = 0.2
    ) -> List[Dict]:
        """Find GFP peaks and extract temporal windows around them."""
        if gfp_values is None:
            gfp_values = self.calculate_gfp(data)
            
        # Convert time parameters to samples
        min_samples = int(min_duration_ms * self.sampling_rate / 1000)
        window_samples = int(window_size_ms * self.sampling_rate / 1000)
        
        # Calculate threshold and prominence
        gfp_max = np.max(gfp_values)
        gfp_threshold = gfp_threshold_pct * gfp_max
        prominence = prominence_pct * gfp_max
        
        # Find peaks
        peaks, properties = find_peaks(
            gfp_values,
            height=gfp_threshold,
            distance=min_samples,
            prominence=prominence
        )
        
        # Extract windows around peaks
        peak_windows = []
        for peak_idx in peaks:
            start_idx = max(0, peak_idx - window_samples)
            end_idx = min(len(gfp_values), peak_idx + window_samples + 1)
            
            window = {
                'peak_idx': int(peak_idx),
                'window_indices': np.arange(start_idx, end_idx).tolist(),
                'gfp_values': gfp_values[start_idx:end_idx].tolist(),
                'peak_gfp': float(gfp_values[peak_idx]),
                'time_ms': float(peak_idx / self.sampling_rate * 1000)
            }
            peak_windows.append(window)
        
        return peak_windows

    def plot_gfp(self, gfp_values: np.ndarray, peaks: List[int] = None, 
                 save_path: Optional[str] = None) -> None:
        """Plot GFP values and optionally mark peaks."""
        plt.figure(figsize=(15, 5))
        plt.plot(gfp_values, 'b-', label='GFP', alpha=0.7)
        
        if peaks is not None:
            plt.plot(peaks, gfp_values[peaks], 'r.', 
                    label=f'Peaks (n={len(peaks)})', markersize=10)
            
        plt.title('Global Field Power')
        plt.xlabel('Time points')
        plt.ylabel('GFP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# The utility functions should be outside the class
def setup_mne_logging():
    """Configure MNE logging to be less verbose"""
    mne.set_log_level('WARNING')
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def preprocess_eeg_data(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply standard preprocessing steps for microstate analysis."""
    raw_proc = raw.copy()
    raw_proc.filter(l_freq=1.0, h_freq=30.0)
    raw_proc.set_eeg_reference('average', projection=True)
    return raw_proc

# Example usage
if __name__ == "__main__":
    # Load EDF file
    edf_path = Path("/Users/sthukral/Documents/GitHub/EEG_MS_MONO/data/1.0.0/Subject00_1.edf")
    
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
        
    # Read EEG data
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data = raw.get_data()
    
    # Initialize calculator with sampling rate from the data
    calculator = GFPCalculator(sampling_rate=raw.info['sfreq'])
    
    # Calculate GFP and find peaks
    gfp_values = calculator.calculate_gfp(data)
    peak_windows = calculator.find_gfp_peaks_and_windows(data, gfp_values)
    
    # Extract peak indices for plotting
    peak_indices = [window['peak_idx'] for window in peak_windows]
    
    # Plot results
    calculator.plot_gfp(gfp_values, peak_indices)
    
    # Print some statistics
    print(f"\nFound {len(peak_windows)} GFP peaks")
    print(f"Average GFP at peaks: {np.mean([w['peak_gfp'] for w in peak_windows]):.8f}")
    print(f"Data duration: {len(gfp_values)/raw.info['sfreq']:.2f} seconds")