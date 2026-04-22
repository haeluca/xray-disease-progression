import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_train_transforms, get_val_transforms


def train_classifier(config, classifier, device="cuda"):
    logger = Logger(config["paths"]["log_dir"])
    logger.save_config(config)

    train_transforms = get_train_transforms(config["data"]["image_size"])
    val_transforms = get_val_transforms(config["data"]["image_size"])

    train_dataset = BaseDataset(
        split_csv=config["data"]["train_split"],
        metadata_csv=config["data"]["metadata_path"],
        roi_dir=config["data"]["roi_dir"],
        transforms=train_transforms,
        image_size=config["data"]["image_size"],
    )

    val_dataset = BaseDataset(
        split_csv=config["data"]["val_split"],
        metadata_csv=config["data"]["metadata_path"],
        roi_dir=config["data"]["roi_dir"],
        transforms=val_transforms,
        image_size=config["data"]["image_size"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    optimizer = optim.AdamW(
        classifier.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"].get("weight_decay", 1e-5),
    )

    criterion = nn.BCEWithLogitsLoss()

    classifier = classifier.to(device)
    classifier.train()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")

        for batch in pbar:
            images = batch["image"].to(device)
            optimizer.zero_grad()

            logits = classifier(images)
            loss = criterion(logits, torch.zeros_like(logits))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        logger.log_scalar("train/loss", avg_train_loss, epoch)

        classifier.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                logits = classifier(images)
                loss = criterion(logits, torch.zeros_like(logits))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.log_scalar("val/loss", avg_val_loss, epoch)

        classifier.train()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                },
                f"{config['paths']['checkpoint_dir']}/epoch_{epoch}.pt",
                is_best=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    logger.close()
    print(f"Training complete. Best checkpoint saved.")
