import torch
import numpy as np


def label_fidelity(predictions, targets):
    predictions = torch.round(torch.sigmoid(predictions)).cpu().numpy()
    targets = targets.cpu().numpy()
    accuracy = np.mean(predictions == targets)
    return accuracy


def per_feature_agreement(predictions, targets):
    predictions = torch.round(torch.sigmoid(predictions)).cpu().numpy()
    targets = targets.cpu().numpy()

    num_features = predictions.shape[1]
    per_feature_acc = []

    for i in range(num_features):
        acc = np.mean(predictions[:, i] == targets[:, i])
        per_feature_acc.append(acc)

    return per_feature_acc


def condition_matching_score(generated_logits, target_features):
    predicted_features = torch.round(torch.sigmoid(generated_logits))
    target_features_binary = torch.round(torch.sigmoid(target_features))

    match_rate = torch.mean((predicted_features == target_features_binary).float()).item()
    return match_rate
