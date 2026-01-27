"""
Semantic segmentation model for pixel-wise classification.

Implements a U-Net style architecture optimized for Akida hardware.
"""
import torch
import torch.nn as nn
from typing import List


class DownBlock(nn.Module):
    """
    Downsampling block for encoder.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through down block.

        Returns:
            Tuple of (pooled output, skip connection)
        """
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block for decoder.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through up block with skip connection.

        Args:
            x: Input from previous layer
            skip: Skip connection from encoder

        Returns:
            Upsampled and processed features
        """
        x = self.up(x)

        # Handle size mismatch
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)

        if diff_h > 0 or diff_w > 0:
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                       diff_h // 2, diff_h - diff_h // 2])

        # Concatenate skip connection
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class SegmentationModel(nn.Module):
    """
    U-Net style segmentation model for semantic segmentation.

    Args:
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of segmentation classes
        base_channels: Base number of channels
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 21,
        base_channels: int = 64
    ):
        super().__init__()

        self.num_classes = num_classes

        # Encoder (downsampling path)
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 16, base_channels * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 16),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling path)
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)

        # Final classification layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segmentation model.

        Args:
            x: Input image tensor (batch, channels, height, width)

        Returns:
            Segmentation map (batch, num_classes, height, width)
        """
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Final classification
        x = self.final_conv(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict segmentation masks.

        Args:
            x: Input image tensor (batch, channels, height, width)

        Returns:
            Predicted class indices (batch, height, width)
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions
