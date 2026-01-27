"""
Vision model for object detection and classification using 2D convolutions.

Optimized for BrainChip's Akida hardware with event-based processing.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class EventBasedConv2d(nn.Module):
    """
    Event-based 2D convolutional block optimized for Akida hardware.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        padding: Padding for convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through event-based conv block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with event-based convolutions.

    Args:
        channels: Number of channels
        stride: Stride for first convolution
    """

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = EventBasedConv2d(channels, channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


class VisionBackbone(nn.Module):
    """
    Vision backbone network for feature extraction.

    Args:
        in_channels: Number of input channels (3 for RGB)
        base_channels: Base number of channels
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            EventBasedConv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Feature extraction stages
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels),
            ResidualBlock(base_channels)
        )

        self.stage2 = nn.Sequential(
            EventBasedConv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2)
        )

        self.stage3 = nn.Sequential(
            EventBasedConv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        self.stage4 = nn.Sequential(
            EventBasedConv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input image tensor (batch, channels, height, width)

        Returns:
            Dictionary of feature maps at different scales
        """
        features = {}

        x = self.stem(x)
        features['stem'] = x

        x = self.stage1(x)
        features['stage1'] = x

        x = self.stage2(x)
        features['stage2'] = x

        x = self.stage3(x)
        features['stage3'] = x

        x = self.stage4(x)
        features['stage4'] = x

        return features


class ObjectClassifier(nn.Module):
    """
    Object classification model using vision backbone.

    Args:
        num_classes: Number of object classes
        in_channels: Number of input image channels
        base_channels: Base number of channels in backbone
    """

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 32
    ):
        super().__init__()

        self.backbone = VisionBackbone(in_channels, base_channels)

        # Global average pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input image tensor (batch, channels, height, width)

        Returns:
            Class logits (batch, num_classes)
        """
        features = self.backbone(x)
        x = features['stage4']
        x = self.classifier(x)
        return x


class ObjectDetector(nn.Module):
    """
    Simple object detection model with bounding box regression.

    Args:
        num_classes: Number of object classes
        in_channels: Number of input image channels
        base_channels: Base number of channels
        num_anchors: Number of anchor boxes per location
    """

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: int = 3,
        base_channels: int = 32,
        num_anchors: int = 9
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = VisionBackbone(in_channels, base_channels)

        # Detection head
        feature_channels = base_channels * 8

        self.class_head = nn.Conv2d(
            feature_channels,
            num_anchors * num_classes,
            kernel_size=1
        )

        self.bbox_head = nn.Conv2d(
            feature_channels,
            num_anchors * 4,
            kernel_size=1
        )

        self.objectness_head = nn.Conv2d(
            feature_channels,
            num_anchors,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for object detection.

        Args:
            x: Input image tensor (batch, channels, height, width)

        Returns:
            Dictionary with 'classes', 'boxes', and 'objectness' predictions
        """
        features = self.backbone(x)
        x = features['stage4']

        # Predictions
        class_pred = self.class_head(x)
        bbox_pred = self.bbox_head(x)
        obj_pred = self.objectness_head(x)

        batch_size = x.size(0)
        h, w = x.size(2), x.size(3)

        # Reshape predictions
        class_pred = class_pred.view(batch_size, self.num_anchors, self.num_classes, h, w)
        class_pred = class_pred.permute(0, 1, 3, 4, 2).contiguous()

        bbox_pred = bbox_pred.view(batch_size, self.num_anchors, 4, h, w)
        bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2).contiguous()

        obj_pred = obj_pred.view(batch_size, self.num_anchors, h, w)

        return {
            'classes': class_pred,
            'boxes': bbox_pred,
            'objectness': obj_pred
        }
