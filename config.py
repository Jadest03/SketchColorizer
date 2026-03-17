import torch

class Config:
    checkpoint_dir = "models"
    result_dir = "results"
    
    # hyperparameters
    image_size = 256
    batch_size = 16
    epochs = 200         
    lr = 3e-4            
    noise_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    cfg_scale = 4.0  
    
    # GPU
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

       