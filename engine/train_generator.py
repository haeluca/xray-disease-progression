import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from datasets.transforms import get_train_transforms, get_val_transforms
from datasets.feature_conditioned_dataset import FeatureConditionedDataset
from datasets.contralateral_dataset import ContralateralDataset


def _build_dataloaders(config, project, feature_schema=None):
    train_tf = get_train_transforms(config["data"]["image_size"])
    val_tf = get_val_transforms(config["data"]["image_size"])
    num_features = len(feature_schema) if feature_schema is not None else config["data"]["num_features"]

    if project == "a":
        train_ds = FeatureConditionedDataset(
            split_csv=config["data"]["train_split"],
            metadata_csv=config["data"]["metadata_path"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=train_tf,
            image_size=config["data"]["image_size"],
            randomize_target=config["training"].get("randomize_target_features", False),
            feature_schema=feature_schema,
        )
        val_ds = FeatureConditionedDataset(
            split_csv=config["data"]["val_split"],
            metadata_csv=config["data"]["metadata_path"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=val_tf,
            image_size=config["data"]["image_size"],
            randomize_target=False,
            feature_schema=feature_schema,
        )
    elif project == "b":
        train_ds = ContralateralDataset(
            split_csv=config["data"]["train_split"],
            contralateral_pairs_csv=config["data"]["contralateral_pairs"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=train_tf,
            image_size=config["data"]["image_size"],
            feature_schema=feature_schema,
        )
        val_ds = ContralateralDataset(
            split_csv=config["data"]["val_split"],
            contralateral_pairs_csv=config["data"]["contralateral_pairs"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=val_tf,
            image_size=config["data"]["image_size"],
            feature_schema=feature_schema,
        )
    else:
        raise ValueError(f"Unknown project: {project}")

    bs = config["training"]["batch_size"]
    nw = config["training"].get("num_workers", 0)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader


def _unpack_batch(batch, project, objective, device):
    if project == "a":
        image = batch["image"].to(device)
        cond_vec = batch["target_features"].to(device)
        if objective == "ddpm":
            return {"target": image, "condition_image": image, "condition_vector": cond_vec}
        return {"target": image, "condition_vector": cond_vec}
    else:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        delta = batch["feature_delta"].to(device)
        return {"target": target, "source": source, "condition_vector": delta}


def _step_ddpm(model, batch_data, project):
    target = batch_data["target"]
    cond_vec = batch_data["condition_vector"]
    if project == "a":
        cond_img = batch_data["condition_image"]
    else:
        cond_img = batch_data["source"]

    B = target.shape[0]
    t = torch.randint(0, model.T, (B,), device=target.device).long()
    loss = model(target, cond_img, t, condition_vector=cond_vec)
    return loss


def _step_vae(model, batch_data):
    target = batch_data["target"]
    cond = batch_data["condition_vector"]
    recon, mu, logvar = model(target, cond)
    recon_loss = nn.functional.mse_loss(recon, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl, {"recon": recon_loss.item(), "kl": kl.item()}


def _step_pix2pix(generator, discriminator, batch_data, opt_g, opt_d, adv_weight=0.1, l1_weight=10.0):
    source = batch_data["source"]
    target = batch_data["target"]

    fake = generator(source)

    opt_d.zero_grad()
    real_pair = torch.cat([source, target], dim=1)
    fake_pair = torch.cat([source, fake.detach()], dim=1)
    d_real = discriminator(real_pair)
    d_fake = discriminator(fake_pair)
    d_loss = 0.5 * (
        nn.functional.mse_loss(d_real, torch.ones_like(d_real))
        + nn.functional.mse_loss(d_fake, torch.zeros_like(d_fake))
    )
    d_loss.backward()
    opt_d.step()

    opt_g.zero_grad()
    fake_pair = torch.cat([source, fake], dim=1)
    d_fake = discriminator(fake_pair)
    g_adv = nn.functional.mse_loss(d_fake, torch.ones_like(d_fake))
    g_l1 = nn.functional.l1_loss(fake, target)
    g_loss = adv_weight * g_adv + l1_weight * g_l1
    g_loss.backward()
    opt_g.step()

    return g_loss, {"g_adv": g_adv.item(), "g_l1": g_l1.item(), "d_loss": d_loss.item()}


def train_generator(config, model, project, objective, device="cuda", discriminator=None, feature_schema=None):
    """
    project: "a" or "b"
    objective: "ddpm" | "vae" | "pix2pix"
    """
    logger = Logger(config["paths"]["log_dir"])
    logger.save_config(config)

    train_loader, val_loader = _build_dataloaders(config, project, feature_schema=feature_schema)

    model = model.to(device)
    params = list(model.parameters())
    lr = float(config["optimizer"]["lr"])
    weight_decay = float(config["optimizer"].get("weight_decay", 1e-5))
    opt = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    opt_d = None
    if objective == "pix2pix":
        if discriminator is None:
            raise ValueError("pix2pix objective requires a discriminator")
        discriminator = discriminator.to(device)
        opt_d = optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    patience = 0
    max_patience = config["training"].get("early_stopping_patience", 30)
    ckpt_dir = config["paths"]["checkpoint_dir"]

    global_step = 0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        if discriminator is not None:
            discriminator.train()

        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{objective}] Epoch {epoch + 1}/{config['training']['epochs']}")
        for batch in pbar:
            batch_data = _unpack_batch(batch, project, objective, device)

            if objective == "ddpm":
                opt.zero_grad()
                loss = _step_ddpm(model, batch_data, project)
                loss.backward()
                opt.step()
                scalar = loss.item()
                logger.log_scalar("train/loss", scalar, global_step)

            elif objective == "vae":
                opt.zero_grad()
                loss, parts = _step_vae(model, batch_data)
                loss.backward()
                opt.step()
                scalar = loss.item()
                logger.log_scalar("train/loss", scalar, global_step)
                for k, v in parts.items():
                    logger.log_scalar(f"train/{k}", v, global_step)

            elif objective == "pix2pix":
                loss, parts = _step_pix2pix(
                    model, discriminator, batch_data, opt, opt_d,
                    adv_weight=config["loss"].get("adversarial_weight", 0.1),
                    l1_weight=config["loss"].get("l1_weight", 10.0),
                )
                scalar = loss.item()
                logger.log_scalar("train/g_loss", scalar, global_step)
                for k, v in parts.items():
                    logger.log_scalar(f"train/{k}", v, global_step)
            else:
                raise ValueError(f"Unknown objective: {objective}")

            train_loss += scalar
            pbar.set_postfix({"loss": f"{scalar:.4f}"})
            global_step += 1

        avg_train = train_loss / max(1, len(train_loader))

        model.eval()
        if discriminator is not None:
            discriminator.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_data = _unpack_batch(batch, project, objective, device)
                if objective == "ddpm":
                    loss = _step_ddpm(model, batch_data, project)
                elif objective == "vae":
                    loss, _ = _step_vae(model, batch_data)
                else:
                    fake = model(batch_data["source"])
                    loss = nn.functional.l1_loss(fake, batch_data["target"])
                val_loss += loss.item()

        avg_val = val_loss / max(1, len(val_loader))
        logger.log_scalar("val/loss", avg_val, epoch)
        print(f"Epoch {epoch + 1}: train={avg_train:.4f} val={avg_val:.4f}")

        is_best = avg_val < best_val
        if is_best:
            best_val = avg_val
            patience = 0
        else:
            patience += 1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss": avg_val,
            "objective": objective,
            "project": project,
        }
        if discriminator is not None:
            state["discriminator_state_dict"] = discriminator.state_dict()
            state["optimizer_d_state_dict"] = opt_d.state_dict()

        save_checkpoint(state, f"{ckpt_dir}/epoch_{epoch}.pt", is_best=is_best)

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    logger.close()
    return best_val
