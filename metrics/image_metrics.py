"""
Pixel-level image quality metrics implemented in pure PyTorch (no skimage dependency).

All public functions expect image tensors in [0, 1] unless noted.
evaluate_image_quality() accepts [-1, 1] and normalises internally.
"""

import torch
import torch.nn.functional as F


def ssim(pred, target, window_size=11, sigma=1.5):
    """
    Structural Similarity Index (SSIM) averaged over the batch and spatial dims.

    Args:
        pred, target: (B, C, H, W) tensors in [0, 1].
        window_size:  Size of the Gaussian smoothing kernel.
        sigma:        Standard deviation of the Gaussian kernel.
    Returns:
        Scalar SSIM in [-1, 1] (1.0 = identical images).
    """
    def gaussian_window(window_size, sigma):
        x = torch.arange(window_size).float() - (window_size - 1) / 2
        gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return gauss / gauss.sum()

    B, C, H, W = pred.shape
    if H < window_size or W < window_size:
        return torch.tensor(0.0, device=pred.device)

    window = gaussian_window(window_size, sigma).to(pred.device)
    window = window.view(1, 1, -1, 1) * window.view(1, 1, 1, -1)
    window = window.repeat(C, 1, 1, 1)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    # stability constants from Wang et al. (2004); L=1 for [0,1] images
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio in dB.

    Args:
        pred, target: (B, C, H, W) tensors in [0, 1].
        max_val:      Maximum possible pixel value (1.0 for normalised images).
    Returns:
        Scalar PSNR in dB (higher = better; inf if images are identical).
    """
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    psnr_val = 20 * torch.log10(torch.tensor(max_val, device=pred.device) / torch.sqrt(mse))
    return psnr_val


def l1_distance(pred, target):
    """Mean absolute pixel error between pred and target."""
    return torch.mean(torch.abs(pred - target))


def evaluate_image_quality(pred, target):
    """
    Compute SSIM, PSNR, and L1 for a batch of images in [-1, 1].

    Normalises both tensors from [-1, 1] → [0, 1] before computing metrics.

    Returns:
        Dict with keys 'ssim', 'psnr', 'l1' (all Python floats).
    """
    pred_01 = (pred * 0.5 + 0.5).clamp(0, 1)
    target_01 = (target * 0.5 + 0.5).clamp(0, 1)
    return {
        "ssim": ssim(pred_01, target_01).item(),
        "psnr": psnr(pred_01, target_01).item(),
        "l1": l1_distance(pred_01, target_01).item(),
    }
