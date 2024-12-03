import os
import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, val_loss, config):
    """Save model checkpoint and upload to wandb"""
    try:
        # Create checkpoints directory if it doesn't exist
        checkpoint_dir = os.path.join(wandb.run.dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint locally
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
        }, checkpoint_path)
        
        # Log checkpoint as artifact in wandb
        artifact = wandb.Artifact(
            name=f'model-checkpoint-epoch-{epoch}',
            type='model',
            description=f'Model checkpoint from epoch {epoch} with validation loss {val_loss:.4f}'
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
        logger.info(f"Saved checkpoint for epoch {epoch} with validation loss {val_loss:.4f}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
        # Continue training even if checkpoint saving fails
        pass

def train_model(model, train_loader, val_loader, config):
    """Training function with improved error handling and logging"""
    try:
        # Set default values for optional parameters
        default_config = {
            'early_stopping_patience': 10,
            'scheduler_patience': 5,
            'scheduler_factor': 0.2,
        }
        
        # Update config with defaults for missing values
        for key, default_value in default_config.items():
            if not hasattr(config, key):
                setattr(config, key, default_value)
                logger.info(f"Using default value for {key}: {default_value}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        wandb.watch(model, log="all", log_freq=100)
        
        # Optimizer setup
        optimizer_class = Adam if config.optimizer.lower() == 'adam' else AdamW
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            verbose=True
        )
        
        criterion = torch.nn.MSELoss()
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(config.num_epochs):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # Extract image tensor from batch dictionary and move to device
                    data = batch['image'].to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reconstructed, latent = model(data)
                    loss = criterion(reconstructed, data)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})
                    
                    # Log batch metrics
                    if batch_idx % 500 == 0:
                        wandb.log({
                            'batch_loss': loss.item(),
                            'epoch': epoch,
                            'batch': batch_idx,
                            'learning_rate': optimizer.param_groups[0]['lr']
                        })
                        
                except RuntimeError as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    data = batch['image'].to(device)
                    reconstructed, latent = model(data)
                    loss = criterion(reconstructed, data)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            # Logging
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                save_checkpoint(model, optimizer, epoch, avg_val_loss, config)
            else:
                early_stopping_counter += 1
                
            if early_stopping_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise