import torch
import numpy as np

from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def evaluate_feature_fidelity(generated_images, target_features, classifier, feature_schema=None, device="cpu"):
    """
    Run frozen classifier on generated images and compare predicted vs target feature vectors.

    Returns a dict with per-feature accuracy (ordinal) or MAE (continuous), plus an overall score.
    generated_images: (B, C, H, W) tensor in [-1, 1]
    target_features:  (B, num_features) tensor with raw label values
    classifier: ClassifierBackbone in eval mode
    """
    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    classifier = classifier.to(device)
    classifier.eval()

    generated_images = generated_images.to(device)
    target_features = target_features.to(device)

    with torch.no_grad():
        outputs = classifier(generated_images)

    results = {}
    total_score = 0.0

    for i, feat in enumerate(feature_schema):
        target_col = target_features[:, i]
        out = outputs[i]

        if feat["type"] == "ordinal":
            pred = out.argmax(dim=1)
            acc = (pred == target_col.long()).float().mean().item()
            results[f"{feat['name']}_acc"] = acc
            total_score += acc
        else:
            pred = out.squeeze(-1)
            mae = (pred - target_col.float()).abs().mean().item()
            results[f"{feat['name']}_mae"] = mae
            # invert MAE to a 0-1 score assuming values in [0,1]
            total_score += max(0.0, 1.0 - mae)

    results["overall_fidelity"] = total_score / len(feature_schema)
    return results


def per_feature_agreement(outputs, targets, feature_schema=None):
    """
    Given raw classifier outputs (list of tensors) and a target tensor (B, num_features),
    return per-feature accuracy / MAE dict.
    """
    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    results = {}
    for i, feat in enumerate(feature_schema):
        target_col = targets[:, i]
        out = outputs[i]
        if feat["type"] == "ordinal":
            pred = out.argmax(dim=1)
            acc = (pred == target_col.long()).float().mean().item()
            results[feat["name"]] = acc
        else:
            pred = out.squeeze(-1)
            mae = (pred - target_col.float()).abs().mean().item()
            results[feat["name"]] = mae
    return results
