import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(pred, target)
    elif mode == "l2" or mode == "mse":
        return F.mse_loss(pred, target)
    elif mode == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    else:
        raise ValueError(f"Unknown loss mode: {mode}")


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def adversarial_loss(disc_output: torch.Tensor, is_real: bool) -> torch.Tensor:
    if is_real:
        target = torch.ones_like(disc_output)
    else:
        target = torch.zeros_like(disc_output)
    return F.binary_cross_entropy_with_logits(disc_output, target)


def condition_consistency_loss(
    generated_image: torch.Tensor,
    target_features: torch.Tensor,
    classifier: nn.Module,
    weight: float = 1.0,
) -> torch.Tensor:
    logits = classifier(generated_image)
    loss = F.binary_cross_entropy_with_logits(logits, target_features)
    return weight * loss
