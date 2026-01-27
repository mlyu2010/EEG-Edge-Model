"""
Training utilities for TENN models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict
from pathlib import Path
from loguru import logger
import time


class Trainer:
    """
    Generic trainer class for PyTorch models.

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda', 'mps', or 'cpu')
        scheduler: Optional learning rate scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        scheduler: Optional[object] = None
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            metric_fn: Optional function to compute metrics

        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)

            # Compute loss
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            if metric_fn is not None:
                all_predictions.append(output.detach())
                all_targets.append(target.detach())

        avg_loss = total_loss / total_samples

        results = {'loss': avg_loss}

        if metric_fn is not None and len(all_predictions) > 0:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            metrics = metric_fn(predictions, targets)
            results.update(metrics)

        return results

    def validate(
        self,
        val_loader: DataLoader,
        metric_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            metric_fn: Optional function to compute metrics

        Returns:
            Dictionary with loss and metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Compute loss
                loss = self.criterion(output, target)

                # Track metrics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                if metric_fn is not None:
                    all_predictions.append(output)
                    all_targets.append(target)

        avg_loss = total_loss / total_samples

        results = {'loss': avg_loss}

        if metric_fn is not None and len(all_predictions) > 0:
            predictions = torch.cat(all_predictions)
            targets = torch.cat(all_targets)
            metrics = metric_fn(predictions, targets)
            results.update(metrics)

        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        metric_fn: Optional[Callable] = None,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 5
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            metric_fn: Optional metric function
            checkpoint_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping

        Returns:
            Dictionary with training history
        """
        best_val_loss = float('inf')
        patience_counter = 0

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Train
            train_results = self.train_epoch(train_loader, metric_fn)
            self.train_losses.append(train_results['loss'])

            log_msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_results['loss']:.4f}"

            # Validate
            if val_loader is not None:
                val_results = self.validate(val_loader, metric_fn)
                self.val_losses.append(val_results['loss'])

                log_msg += f" - Val Loss: {val_results['loss']:.4f}"

                # Early stopping
                if val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    patience_counter = 0

                    # Save best model
                    if checkpoint_dir:
                        best_model_path = Path(checkpoint_dir) / "best_model.pt"
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': best_val_loss,
                        }, best_model_path)
                        logger.info(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time
            log_msg += f" - Time: {epoch_time:.2f}s"

            logger.info(log_msg)

            # Save periodic checkpoint
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)

        logger.info("Training completed!")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

    def save_model(self, path: str):
        """
        Save model state dict.

        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model state dict.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


def compute_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels

    Returns:
        Dictionary with metrics
    """
    # Get predicted classes
    _, pred_classes = predictions.max(1)

    # Accuracy
    correct = (pred_classes == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def compute_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dictionary with metrics
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    return {
        'mse': mse,
        'mae': mae,
        'rmse': mse ** 0.5
    }
