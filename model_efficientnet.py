# model_efficientnet.py
import torch
import torch.nn as nn
from torchvision import models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # load torchvision EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        # replace classifier
        in_features = self.backbone.classifier[1].in_features
        # simple head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
