# EEG Microstate Analysis Project

This project implements a deep learning approach to EEG microstate analysis using convolutional autoencoders and various clustering techniques.

## Project Structure

The project is organized into several key components:
- `configs/`: Configuration files for different pipeline stages
- `models/`: Neural network and clustering model implementations
- `utils/`: Utility functions for data processing and visualization
- `scripts/`: Execution scripts for different pipeline stages
- `tests/`: Unit tests
- `outputs/`: Generated outputs (topomaps, trained models, clustering results)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Pipeline Stages

### 1. Topographic Map Generation
- Converts raw EEG data into 64x64 topographic maps
- Supports both GFP peak-based and full dataset approaches

### 2. Convolutional Autoencoder (CAE) Training
- Reduces dimensionality of topographic maps
- Uses WandB for experiment tracking

### 3. Clustering Analysis
- Implements multiple clustering approaches
- Evaluates cluster quality using domain-specific metrics

## Usage

1. Generate topographic maps:
```bash
python scripts/generate_topomaps.py
```

2. Train the CAE:
```bash
python scripts/train_cae.py
```

3. Run clustering analysis:
```bash
python scripts/run_clustering.py
```

## Testing

Run tests using pytest:
```bash
pytest tests/
``` 