import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from datasets.base_dataset import BaseDataset
from datasets.transforms import get_train_transforms, get_val_transforms


def _compute_loss(outputs, targets, feature_schema):
    total = 0.0
    parts = {}
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for i, feat in enumerate(feature_schema):
        target_col = targets[:, i]
        if feat["type"] == "ordinal":
            loss_i = ce(outputs[i], target_col.long())
        else:
            loss_i = mse(outputs[i].squeeze(-1), target_col.float())
        total = total + loss_i
        parts[feat["name"]] = loss_i.item()
    return total, parts


def _compute_accuracy(outputs, targets, feature_schema):
    metrics = {}
    for i, feat in enumerate(feature_schema):
        target_col = targets[:, i]
        if feat["type"] == "ordinal":
            pred = outputs[i].argmax(dim=1)
            acc = (pred == target_col.long()).float().mean().item()
            metrics[f"{feat['name']}_acc"] = acc
        else:
            mae = (outputs[i].squeeze(-1) - target_col.float()).abs().mean().item()
            metrics[f"{feat['name']}_mae"] = mae
    return metrics


def train_classifier(config, classifier, device="cuda"):
    logger = Logger(config["paths"]["log_dir"])
    logger.save_config(config)

    train_transforms = get_train_transforms(config["data"]["image_size"])
    val_transforms = get_val_transforms(config["data"]["image_size"])

    feature_schema = classifier.feature_schema
    num_features = len(feature_schema)

    train_dataset = BaseDataset(
        split_csv=config["data"]["train_split"],
        metadata_csv=config["data"]["metadata_path"],
        roi_dir=config["data"]["roi_dir"],
        transforms=train_transforms,
        image_size=config["data"]["image_size"],
        num_features=num_features,
        feature_schema=feature_schema,
    )

    val_dataset = BaseDataset(
        split_csv=config["data"]["val_split"],
        metadata_csv=config["data"]["metadata_path"],
        roi_dir=config["data"]["roi_dir"],
        transforms=val_transforms,
        image_size=config["data"]["image_size"],
        num_features=num_features,
        feature_schema=feature_schema,
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
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"].get("weight_decay", 1e-5)),
    )

    classifier = classifier.to(device)
    classifier.train()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config["training"]["epochs"]):
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['training']['epochs']}")

        for batch in pbar:
            images = batch["image"].to(device)
            targets = batch["features"].to(device)
            optimizer.zero_grad()

            outputs = classifier(images)
            loss, _ = _compute_loss(outputs, targets, feature_schema)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(1, len(train_loader))
        logger.log_scalar("train/loss", avg_train_loss, epoch)

        classifier.eval()
        val_loss = 0.0
        agg_metrics = {}

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["features"].to(device)
                outputs = classifier(images)
                loss, _ = _compute_loss(outputs, targets, feature_schema)
                val_loss += loss.item()

                m = _compute_accuracy(outputs, targets, feature_schema)
                for k, v in m.items():
                    agg_metrics.setdefault(k, []).append(v)

        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.log_scalar("val/loss", avg_val_loss, epoch)
        for k, vs in agg_metrics.items():
            logger.log_scalar(f"val/{k}", sum(vs) / len(vs), epoch)

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
                    "feature_schema": feature_schema,
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
