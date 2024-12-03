import os
import json
import numpy as np
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from scipy.signal import find_peaks

class EEGProcessor:
    def __init__(self, data_dir, save_dir, use_gfp_peaks=False, gfp_threshold=0.5, map_interval_ms=40):
        # Set up logging
        self._setup_logging()
        
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.sampling_rate = 500  # Hz
        
        # 40ms interval based on literature (Michel & Koenig, 2018)
        self.map_interval_ms = map_interval_ms
        
        # Create output directories
        self.resting_dir = self.save_dir / 'resting_state'
        self.task_dir = self.save_dir / 'task_state'
        self.resting_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        
        self.channel_mapping = {
            'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2',
            'EEG F3': 'F3', 'EEG F4': 'F4',
            'EEG F7': 'F7', 'EEG F8': 'F8',
            'EEG T3': 'T7', 'EEG T4': 'T8',  
            'EEG C3': 'C3', 'EEG C4': 'C4',
            'EEG T5': 'P7', 'EEG T6': 'P8',  
            'EEG P3': 'P3', 'EEG P4': 'P4',
            'EEG O1': 'O1', 'EEG O2': 'O2',
            'EEG Fz': 'Fz', 'EEG Cz': 'Cz', 'EEG Pz': 'Pz',
        }
        
        # Set up visualization parameters
        self._setup_visualization()
        
        self.use_gfp_peaks = use_gfp_peaks
        self.gfp_threshold = gfp_threshold

    def _setup_logging(self):
        """Configure logging with detailed formatting"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"eeg_processing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_visualization(self):
        """Configure visualization parameters for consistent output"""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'image.cmap': 'RdBu_r',
            'figure.dpi': 128,
        })

    def validate_eeg_data(self, raw):
        """Validate EEG data quality and configuration"""
        # Check number of channels
        n_channels = len(raw.ch_names)
        expected_channels = len(self.channel_mapping)
        if n_channels < expected_channels:
            raise ValueError(f"Expected {expected_channels} channels, found {n_channels}")
        
        # Check sampling rate
        if raw.info['sfreq'] != self.sampling_rate:
            raise ValueError(f"Expected sampling rate {self.sampling_rate}Hz, found {raw.info['sfreq']}Hz")
        
        # Basic signal quality checks
        data = raw.get_data()
        
    def create_topomap(self, snapshot, eeg_info, output_path):
        """Create and save a single topomap efficiently"""
        fig = plt.figure(figsize=(1, 1), dpi=128)
        ax = fig.add_axes([0, 0, 1, 1])
        
        mne.viz.plot_topomap(
            snapshot, 
            eeg_info,
            axes=ax,
            show=False,
            contours=0,  # No contour lines
            sensors=False,  # Don't show sensors
            outlines='head',
            extrapolate='head',
        )
        
        # Save directly to the correct format
        fig.savefig(
            output_path,
            dpi=128,
            format='png',
            facecolor='white',
            bbox_inches=None,
            pad_inches=0
        )
        plt.close(fig)

    def _verify_image(self, image_path):
        """Simple verification of image existence"""
        if not os.path.exists(image_path):
            raise ValueError(f"Image was not created at {image_path}")

    def calculate_gfp(self, data):
        """Calculate Global Field Power"""
        gfp = np.std(data, axis=0)
        
        self.logger.info(f"GFP stats - Mean: {gfp.mean():.2f}, STD: {gfp.std():.2f}, "
                        f"Min: {gfp.min():.2f}, Max: {gfp.max():.2f}")
        return gfp

    def find_gfp_peaks(self, gfp_values):
        """
        Find GFP peaks representing moments of topographic stability.
        """
        # Parameters for peak detection
        min_duration_ms = 40.0
        gfp_threshold_pct = 0.3
        prominence_pct = 0.2
        
        min_samples = int(min_duration_ms * self.sampling_rate / 1000)
        
        # Calculate adaptive thresholds
        gfp_threshold = np.mean(gfp_values) + gfp_threshold_pct * np.std(gfp_values)
        prominence = prominence_pct * np.std(gfp_values)
        
        # Find peaks
        peaks, _ = find_peaks(
            gfp_values,
            height=gfp_threshold,
            distance=min_samples,
            prominence=prominence
        )
        
        self.logger.info(f"Found {len(peaks)} GFP peaks")
        self.logger.info(f"GFP threshold: {gfp_threshold:.3f}, Prominence: {prominence:.3f}")
        
        return peaks

    def process_gfp_peaks(self, data, eeg_info, peaks, output_dir):
        """Process GFP peaks and create topomaps"""
        self.logger.info(f"Processing {len(peaks)} GFP peaks")
        
        # Initialize metadata
        peak_metadata = []
        
        for i, peak_idx in enumerate(tqdm(peaks, desc="Processing GFP peaks")):
            snapshot = data[:, peak_idx]
            output_path = output_dir / f'topo_peak_{i:04d}.png'
            self.create_topomap(snapshot, eeg_info, output_path)
            
            peak_metadata.append({
                'peak_id': i,
                'peak_index': int(peak_idx),
                'time_ms': float(peak_idx * 1000 / self.sampling_rate)
            })
        
        # Save metadata
        metadata_path = output_dir / 'subject_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_peaks': len(peaks),
                'peaks': peak_metadata
            }, f, indent=2)

    def process_continuous_data(self, data, eeg_info, output_dir):
        """Process continuous EEG data at fixed intervals"""
        self.logger.info("Processing continuous data at fixed intervals")
        
        # Calculate number of samples per interval
        samples_per_interval = int(self.map_interval_ms * self.sampling_rate / 1000)
        total_samples = data.shape[1]
        
        # Generate indices for all timepoints at fixed intervals
        timepoints = range(0, total_samples, samples_per_interval)
        
        # Initialize metadata
        timepoint_metadata = []
        
        for i, tp_idx in enumerate(tqdm(timepoints, desc="Processing continuous data")):
            snapshot = data[:, tp_idx]
            output_path = output_dir / f'topo_continuous_{i:04d}.png'
            self.create_topomap(snapshot, eeg_info, output_path)
            
            timepoint_metadata.append({
                'timepoint_id': i,
                'sample_index': int(tp_idx),
                'time_ms': float(tp_idx * 1000 / self.sampling_rate)
            })
        
        # Save metadata
        metadata_path = output_dir / 'continuous_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'total_timepoints': len(timepoints),
                'interval_ms': self.map_interval_ms,
                'timepoints': timepoint_metadata
            }, f, indent=2)

    def process_eeg(self, file_path, output_dirs, process_mode='both'):
        """Process a single EEG file and generate topomaps"""
        try:
            self.logger.info(f"Processing {file_path} in {process_mode} mode")
            
            # Read and prepare EEG data
            raw = mne.io.read_raw_edf(file_path, preload=True)
            self.validate_eeg_data(raw)
            
            # Drop unnecessary channels
            channels_to_drop = ['ECG ECG', 'EEG A2-A1']
            existing_channels = [ch for ch in channels_to_drop if ch in raw.ch_names]
            if existing_channels:
                raw.drop_channels(existing_channels)
            
            # Set up montage
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.rename_channels(self.channel_mapping)
            raw.set_montage(montage)
            
            # Get EEG data
            eeg_data = raw.copy().pick(["eeg"]).load_data()
            data = eeg_data.get_data()
            
            self.logger.info(f"Data shape: {data.shape}, Duration: {data.shape[1]/self.sampling_rate:.2f}s")
            
            if process_mode in ['peaks', 'both']:
                # Process using GFP peaks
                gfp = self.calculate_gfp(data)
                peaks = self.find_gfp_peaks(gfp)
                self.process_gfp_peaks(data, eeg_data.info, peaks, output_dirs['peaks'])
                self.logger.info(f"Completed processing {len(peaks)} GFP peaks")
            
            if process_mode in ['continuous', 'both']:
                # Process continuous data
                self.process_continuous_data(data, eeg_data.info, output_dirs['continuous'])
                self.logger.info("Completed processing continuous data")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return False
    


def main(local=False, process_mode='both', map_interval_ms=40):
    # Setup directories based on environment
    if local:
        base_dir = Path('/Users/sthukral/Documents/GitHub/EEG_MS_MONO/data')
        data_dir = base_dir / '1.0.0'
        output_base = Path('/Users/sthukral/Documents/GitHub/EEG_MS_MONO/outputs/topomaps/local_run')
        max_files = 2
    else:
        base_dir = Path('/home/CAMPUS/d18129674/EEG_DC_TOPO')
        data_dir = base_dir / 'EEG_data'
        output_base = base_dir / 'TOPOMAPS_OUT_3'
        max_files = None
    
    # Create separate output directories for continuous and peaks
    output_dirs = {
        'continuous': output_base / 'continuous_maps',
        'peaks': output_base / 'gfp_peaks_maps'
    }
    
    # Clear and create output directories based on process_mode
    for mode in ['continuous', 'peaks']:
        if mode in process_mode or process_mode == 'both':
            output_dir = output_dirs[mode]
            if output_dir.exists():
                import shutil
                print(f"Clearing existing {mode} output directory...")
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create state subdirectories
            (output_dir / 'resting_state').mkdir(exist_ok=True)
            (output_dir / 'task_state').mkdir(exist_ok=True)
    
    # Initialize processor
    processor = EEGProcessor(
        data_dir, 
        output_base,
        use_gfp_peaks=True,
        map_interval_ms=map_interval_ms
    )
    
    # Get EDF files
    edf_files = sorted(list(data_dir.glob('Subject*_[12].edf')))
    if local and max_files:
        edf_files = edf_files[:max_files]
        print(f"\nRunning in local mode. Processing {max_files} files.")
    print(f"Found {len(edf_files)} EDF files to process")
    
    # Process count
    successful = 0
    failed = 0
    
    # Process files
    for edf_file in edf_files:
        subject_name = edf_file.stem
        state = 'resting_state' if subject_name.endswith('_1') else 'task_state'
        
        # Create subject directories in appropriate output locations
        subject_dirs = {}
        if process_mode in ['continuous', 'both']:
            subject_dirs['continuous'] = output_dirs['continuous'] / state / subject_name
            subject_dirs['continuous'].mkdir(parents=True, exist_ok=True)
            
        if process_mode in ['peaks', 'both']:
            subject_dirs['peaks'] = output_dirs['peaks'] / state / subject_name
            subject_dirs['peaks'].mkdir(parents=True, exist_ok=True)
        
        if processor.process_eeg(edf_file, subject_dirs, process_mode=process_mode):
            successful += 1
        else:
            failed += 1
            print(f"Failed to process {edf_file.name}")
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")
    print(f"Total files processed: {len(edf_files)}")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process EEG data for topomaps')
    parser.add_argument('--local', action='store_true', help='Run with local paths')
    parser.add_argument('--mode', choices=['peaks', 'continuous', 'both'], 
                       default='both', help='Processing mode for topomaps')
    parser.add_argument('--interval', type=int, default=40,
                       help='Interval in milliseconds for continuous topomaps')
    args = parser.parse_args()
    
    main(local=args.local, process_mode=args.mode, map_interval_ms=args.interval)