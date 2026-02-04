# src/models/resnet_model.py

import torch.nn as nn
from torchvision import models

class ResNetOCT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify first layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final classification layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)
