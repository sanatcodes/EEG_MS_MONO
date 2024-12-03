import pytest
import torch
from models.cae import ConvolutionalAutoencoder

def test_cae_dimensions():
    """Test if CAE maintains expected dimensions."""
    batch_size = 32
    channels = 1
    height = width = 64
    latent_dim = 32
    
    # Create model and dummy data
    model = ConvolutionalAutoencoder(latent_dim=latent_dim)
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    reconstructed, latent = model(x)
    
    # Check dimensions
    assert reconstructed.shape == x.shape, "Reconstruction should match input dimensions"
    assert latent.shape == (batch_size, latent_dim), "Latent space dimension mismatch"
    
def test_cae_encode_decode():
    """Test if encode and decode methods work as expected."""
    model = ConvolutionalAutoencoder()
    x = torch.randn(1, 1, 64, 64)
    
    # Test encode
    latent = model.encode(x)
    assert isinstance(latent, torch.Tensor), "Encode should return a tensor"
    
    # Test decode
    decoded = model.decode(latent)
    assert isinstance(decoded, torch.Tensor), "Decode should return a tensor"
    assert decoded.shape == x.shape, "Decoded output should match input shape" 