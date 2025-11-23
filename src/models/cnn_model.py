import torch
import torch.nn as nn
from torchvision import models

class COVIDxCNN(nn.Module):
    """
    CNN for COVIDx classification.
    Based on ResNet18 with modified final layer.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(COVIDxCNN, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Modify first conv for grayscale (if needed)
        # self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify final layer for our classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for faster FL experiments.
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Test script
model = COVIDxCNN(num_classes=2)
print(f"Model parameters: {model.get_num_parameters():,}")

# Test forward pass
x = torch.randn(4, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")