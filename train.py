import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    CHECKPOINT_EVERY, TRAIN_DATA_PATH, MODEL_SAVE_PATH, T
)
from data.dataset import PairedXrayDataset
from models.unet import UNet
from models.diffusion import DDPM

def train():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = PairedXrayDataset(TRAIN_DATA_PATH)
    if len(dataset) == 0:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_PATH}. Expected structure: {TRAIN_DATA_PATH}/patient_id/t0.png and t1.png")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    unet = UNet(in_channels=2, out_channels=1).to(DEVICE)
    model = DDPM(unet).to(DEVICE)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in progress_bar:
            x_t0 = batch["x_t0"].to(DEVICE)
            x_t1 = batch["x_t1"].to(DEVICE)

            # Sample random timesteps
            t = torch.randint(0, T, (x_t1.shape[0],), device=DEVICE)

            # Compute loss
            loss = model(x_t1, x_t0, t)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
