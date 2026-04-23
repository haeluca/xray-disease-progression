"""
Multi-head ResNet-18 classifier for CMC I OA feature prediction.

Adapted for single-channel (grayscale) X-ray input. The shared ResNet-18 trunk
produces a 512-d feature vector; a small linear trunk compresses it to 256-d;
then one independent head per feature in the schema predicts either a class
distribution (ordinal) or a scalar value (continuous).

Output of forward(): list of tensors, one per feature, in schema order.
  - ordinal feature  → (B, num_classes) logits (use argmax or cross-entropy)
  - continuous feature → (B, 1) scalar (use MSE or MAE)
"""

import torch
import torch.nn as nn
import torchvision.models as models


from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


class ClassifierBackbone(nn.Module):
    """
    Schema-driven multi-head ResNet-18 for OA feature prediction.

    Args:
        feature_schema: List of feature dicts with 'name', 'type', and (for
                        ordinal features) 'num_classes'. Defaults to DEFAULT_FEATURE_SCHEMA.
        num_features:   Ignored if feature_schema is provided; used to build a
                        generic continuous schema when no schema is available.
        pretrained:     Load ImageNet weights for the ResNet trunk (False by default
                        because input is single-channel grayscale, not RGB).
    """

    def __init__(self, feature_schema=None, num_features: int = None, pretrained: bool = False):
        super().__init__()
        if feature_schema is None:
            if num_features is not None:
                feature_schema = [{"name": f"feature_{i}", "type": "continuous"} for i in range(num_features)]
            else:
                feature_schema = DEFAULT_FEATURE_SCHEMA

        self.feature_schema = feature_schema
        self.num_features = len(feature_schema)

        resnet = models.resnet18(pretrained=pretrained)
        # Replace the standard 3-channel conv1 with a 1-channel equivalent
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        self.trunk = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # One linear head per feature; output dim depends on feature type
        self.heads = nn.ModuleList()
        self.head_output_dims = []
        for feat in feature_schema:
            if feat["type"] == "ordinal":
                out_dim = feat["num_classes"]
            elif feat["type"] == "continuous":
                out_dim = 1
            else:
                raise ValueError(f"Unknown feature type: {feat['type']}")
            self.heads.append(nn.Linear(256, out_dim))
            self.head_output_dims.append(out_dim)

    def _trunk_forward(self, x):
        """Run the shared ResNet-18 backbone and return the 256-d trunk embedding."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        return self.trunk(x)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, H, W) grayscale X-ray tensor in [-1, 1].
        Returns:
            List of tensors in feature_schema order:
              - ordinal  → (B, num_classes) logits
              - continuous → (B, 1) scalar predictions
        """
        h = self._trunk_forward(x)
        return [head(h) for head in self.heads]

    def predict_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single (B, num_features) feature vector per image (argmax for ordinal, scalar for continuous)."""
        outs = self.forward(x)
        pieces = []
        for feat, out in zip(self.feature_schema, outs):
            if feat["type"] == "ordinal":
                pieces.append(out.argmax(dim=1, keepdim=True).float())
            else:
                pieces.append(out)
        return torch.cat(pieces, dim=1)

    def freeze(self):
        """Freeze all parameters — used when the classifier serves as a fixed evaluation backbone."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
