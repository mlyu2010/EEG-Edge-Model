"""
TENN (Temporal Event-based Neural Network) model for EEG signal processing.

This module implements a state-space recurrent model optimized for
1D time series analysis of EEG data on BrainChip's Akida hardware.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with event-based processing.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        dilation: Dilation factor for temporal receptive field
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal block.

        Args:
            x: Input tensor of shape (batch, channels, time)

        Returns:
            Output tensor of shape (batch, channels, time)
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Remove extra padding for causal convolution
        out = out[:, :, :x.size(2)]

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.relu2(out)
        out = self.dropout2(out)

        return out


class StateSpaceLayer(nn.Module):
    """
    State-space recurrent layer for temporal modeling.

    This layer implements a linear state-space model:
    x_t = Ax_{t-1} + Bu_t
    y_t = Cx_t + Du_t

    Args:
        input_dim: Dimension of input features
        state_dim: Dimension of hidden state
        output_dim: Dimension of output features
    """

    def __init__(self, input_dim: int, state_dim: int, output_dim: int):
        super().__init__()

        self.state_dim = state_dim

        # State transition matrices
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through state-space layer.

        Args:
            x: Input tensor of shape (batch, time, features)
            state: Optional initial state of shape (batch, state_dim)

        Returns:
            Tuple of (output, final_state)
        """
        batch_size, seq_len, _ = x.shape

        if state is None:
            state = torch.zeros(batch_size, self.state_dim, device=x.device)

        outputs = []

        for t in range(seq_len):
            u_t = x[:, t, :]

            # State update: x_t = Ax_{t-1} + Bu_t
            state = torch.matmul(state, self.A.t()) + torch.matmul(u_t, self.B.t())

            # Output: y_t = Cx_t + Du_t
            y_t = torch.matmul(state, self.C.t()) + torch.matmul(u_t, self.D.t())
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)

        return output, state


class TENN_EEG(nn.Module):
    """
    TENN model for EEG signal classification.

    This model combines temporal convolutional blocks with state-space
    recurrent layers for efficient processing of EEG time series data.

    Args:
        num_channels: Number of EEG channels
        num_classes: Number of output classes
        temporal_channels: List of channel sizes for temporal blocks
        state_dim: Dimension of state-space hidden state
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_channels: int = 64,
        num_classes: int = 4,
        temporal_channels: list = [32, 64, 128],
        state_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes

        # Temporal convolutional layers
        layers = []
        in_ch = num_channels

        for i, out_ch in enumerate(temporal_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_ch = out_ch

        self.temporal_net = nn.Sequential(*layers)

        # State-space recurrent layer
        self.state_space = StateSpaceLayer(
            input_dim=temporal_channels[-1],
            state_dim=state_dim,
            output_dim=state_dim
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TENN model.

        Args:
            x: Input EEG tensor of shape (batch, channels, time)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Temporal convolution
        x = self.temporal_net(x)

        # Transpose for state-space layer: (batch, time, features)
        x = x.transpose(1, 2)

        # State-space processing
        x, _ = self.state_space(x)

        # Global average pooling over time
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings without classification.

        Args:
            x: Input EEG tensor of shape (batch, channels, time)

        Returns:
            Feature embeddings of shape (batch, state_dim)
        """
        x = self.temporal_net(x)
        x = x.transpose(1, 2)
        x, _ = self.state_space(x)
        x = x.mean(dim=1)
        return x
