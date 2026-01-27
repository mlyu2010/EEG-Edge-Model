#!/usr/bin/env python3
"""
Training script for TENN EEG model.

Usage:
    python scripts/train_eeg_model.py --epochs 100 --batch-size 32
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from app.models.tenn_eeg import TENN_EEG
from app.utils.training import Trainer, compute_classification_metrics
from app.utils.device import get_device, print_device_info
from app.core.config import settings


def create_dummy_dataset(num_samples=1000, num_channels=64, seq_length=256, num_classes=4):
    """
    Create dummy EEG dataset for demonstration.

    In production, replace this with actual EEG data loading.
    """
    X = torch.randn(num_samples, num_channels, seq_length)
    y = torch.randint(0, num_classes, (num_samples,))

    return TensorDataset(X, y)


def main():
    parser = argparse.ArgumentParser(description="Train TENN EEG model")
    parser.add_argument("--epochs", type=int, default=settings.epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=settings.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=settings.learning_rate, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use (auto, cpu, cuda, or mps)")
    parser.add_argument("--num-channels", type=int, default=64, help="Number of EEG channels")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--output-dir", type=str, default="./models/trained", help="Output directory")

    args = parser.parse_args()

    # Print device information
    print_device_info()

    # Get the best available device
    device = get_device(args.device)

    print(f"Training TENN EEG model on {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")

    # Create datasets
    train_dataset = create_dummy_dataset(num_samples=800, num_channels=args.num_channels, num_classes=args.num_classes)
    val_dataset = create_dummy_dataset(num_samples=200, num_channels=args.num_channels, num_classes=args.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = TENN_EEG(
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        temporal_channels=[32, 64, 128],
        state_dim=64
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        metric_fn=compute_classification_metrics,
        checkpoint_dir=args.output_dir,
        early_stopping_patience=10
    )

    # Save final model
    output_path = Path(args.output_dir) / "tenn_eeg_final.pt"
    trainer.save_model(str(output_path))

    print(f"\nTraining complete! Model saved to {output_path}")
    print(f"Best validation loss: {min(history['val_losses']):.4f}")


if __name__ == "__main__":
    main()
