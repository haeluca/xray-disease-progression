import torch

# Image configuration
IMAGE_SIZE = 256
CHANNELS = 1  # Grayscale X-ray
CONDITION_CHANNELS = 1  # Baseline X-ray

# Diffusion configuration
T = 1000  # Number of diffusion timesteps
BETA_START = 1e-4
BETA_END = 0.02

# Training configuration
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 200
CHECKPOINT_EVERY = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
TRAIN_DATA_PATH = "./data/train"
MODEL_SAVE_PATH = "./checkpoints"
