"""
Feature fidelity metrics: measure how well generated images match target OA features
by running the frozen classifier on the generated outputs and comparing predictions
to the requested target feature vector.

Ordinal features (e.g. JSN grade 0-3): evaluated by exact-match accuracy.
Continuous features (e.g. subluxation ratio): evaluated by mean absolute error,
then inverted to a 0-1 score for the overall fidelity scalar.
"""

import torch

from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def evaluate_feature_fidelity(generated_images, target_features, classifier, feature_schema=None, device="cpu"):
    """
    Run frozen classifier on generated images and compare predicted vs target feature vectors.

    Args:
        generated_images: (B, C, H, W) tensor in [-1, 1].
        target_features:  (B, num_features) tensor with raw label values.
        classifier:       ClassifierBackbone — should already be in eval mode and frozen.
        feature_schema:   Feature definition list; defaults to DEFAULT_FEATURE_SCHEMA.
        device:           Device to move tensors to.

    Returns:
        Dict with per-feature metrics and 'overall_fidelity' (mean score in [0, 1]):
          - ordinal features  → '{name}_acc'  (exact-match accuracy)
          - continuous features → '{name}_mae' (mean absolute error)
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
            # invert MAE to a 0-1 score assuming values in [0, 1]
            total_score += max(0.0, 1.0 - mae)

    results["overall_fidelity"] = total_score / len(feature_schema)
    return results


def per_feature_agreement(outputs, targets, feature_schema=None):
    """
    Lightweight agreement check given pre-computed classifier outputs.

    Args:
        outputs: List of tensors (one per feature) as returned by ClassifierBackbone.forward().
        targets: (B, num_features) target label tensor.
        feature_schema: Feature definition list; defaults to DEFAULT_FEATURE_SCHEMA.

    Returns:
        Dict mapping feature name → accuracy (ordinal) or MAE (continuous).
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
