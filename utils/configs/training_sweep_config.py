SWEEP_CONFIG = {
    'method': 'bayes',
    'name': 'simplified-microstate-cae-sweep',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Adjusted for smaller architecture
        'latent_dim': {
            'values': [8, 16, 32]  # Increased values for better representation
        },
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-3
        },
        'batch_size': {
            'values': [32, 64, 128]  # Can use larger batches with smaller images
        },
        'optimizer': {
            'value': 'adam'
        },
        'weight_decay': {
            'values': [1e-5, 1e-4, 1e-3]  # Added as sweep parameter
        },
        'scheduler_patience': {
            'value': 5
        },
        'scheduler_factor': {
            'value': 0.5  # Increased from 0.2 for smoother learning rate decay
        },
        'num_epochs': {
            'value': 50  # Increased to allow for better convergence
        },
        'early_stopping_patience': {
            'value': 5  # Increased to match longer training
        }
    }
}