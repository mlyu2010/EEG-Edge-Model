"""
Anomaly detection model for 1D time series data.

Implements an autoencoder-based approach with TENN components
for detecting anomalies in time series data.
"""
import torch
import torch.nn as nn
from typing import Tuple


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for time series data.

    Args:
        input_channels: Number of input channels
        latent_dim: Dimension of latent representation
        hidden_channels: List of hidden channel dimensions
    """

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        hidden_channels: list = [32, 64, 128]
    ):
        super().__init__()

        layers = []
        in_ch = input_channels

        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        # Final projection to latent space
        self.fc = nn.Linear(hidden_channels[-1] * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to latent representation.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Latent representation (batch, latent_dim)
        """
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TemporalDecoder(nn.Module):
    """
    Temporal decoder for time series reconstruction.

    Args:
        latent_dim: Dimension of latent representation
        output_channels: Number of output channels
        output_length: Length of output sequence
        hidden_channels: List of hidden channel dimensions (reversed from encoder)
    """

    def __init__(
        self,
        latent_dim: int,
        output_channels: int,
        output_length: int,
        hidden_channels: list = [128, 64, 32]
    ):
        super().__init__()

        self.output_length = output_length

        # Project from latent space
        self.fc = nn.Linear(latent_dim, hidden_channels[0] * 8)

        layers = []
        in_ch = hidden_channels[0]

        for out_ch in hidden_channels[1:]:
            layers.extend([
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            in_ch = out_ch

        self.decoder = nn.Sequential(*layers)

        # Final reconstruction layer
        self.final_conv = nn.ConvTranspose1d(
            hidden_channels[-1],
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to time series.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed time series (batch, channels, time)
        """
        x = self.fc(z)
        x = x.view(x.size(0), -1, 8)
        x = self.decoder(x)
        x = self.final_conv(x)

        # Adjust to target length if needed
        if x.size(2) != self.output_length:
            x = nn.functional.interpolate(x, size=self.output_length, mode='linear', align_corners=False)

        return x


class AnomalyDetectionModel(nn.Module):
    """
    Autoencoder-based anomaly detection for time series.

    The model learns to reconstruct normal patterns. Anomalies are detected
    by high reconstruction errors.

    Args:
        input_channels: Number of input channels
        sequence_length: Length of input sequences
        latent_dim: Dimension of latent space
        hidden_channels: List of hidden channel dimensions
    """

    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 256,
        latent_dim: int = 64,
        hidden_channels: list = [32, 64, 128]
    ):
        super().__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim

        self.encoder = TemporalEncoder(
            input_channels,
            latent_dim,
            hidden_channels
        )

        self.decoder = TemporalDecoder(
            latent_dim,
            input_channels,
            sequence_length,
            hidden_channels[::-1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input time series (batch, channels, time)

        Returns:
            Tuple of (reconstruction, latent_representation)
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode time series to latent space.

        Args:
            x: Input time series (batch, channels, time)

        Returns:
            Latent representation (batch, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to time series.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Reconstructed time series (batch, channels, time)
        """
        return self.decoder(z)

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores based on reconstruction error.

        Args:
            x: Input time series (batch, channels, time)

        Returns:
            Anomaly scores (batch,) - higher values indicate more anomalous
        """
        x_recon, _ = self.forward(x)

        # Compute MSE reconstruction error
        reconstruction_error = torch.mean((x - x_recon) ** 2, dim=(1, 2))

        return reconstruction_error

    def detect_anomalies(
        self,
        x: torch.Tensor,
        threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect anomalies based on reconstruction error threshold.

        Args:
            x: Input time series (batch, channels, time)
            threshold: Threshold for anomaly detection

        Returns:
            Tuple of (is_anomaly, anomaly_scores)
            is_anomaly: Boolean tensor (batch,) indicating anomalies
            anomaly_scores: Anomaly scores (batch,)
        """
        scores = self.compute_anomaly_score(x)
        is_anomaly = scores > threshold

        return is_anomaly, scores
