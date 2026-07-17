#!/usr/bin/env python3
"""
Example: Train fused model (AE + Fusion + Transformer).
Run with: python examples/train_fused.py --ae convolutional --transformer stack --epochs 5
"""

import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Add parent to path
sys.path.insert(0, ".")

from hyperkahler.fusion import ModelCombo, HyperkahlerConfig
from hyperkahler.training import UnifiedTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_mnist_loaders(batch_size=32, val_split=0.1):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # Split train into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_data, val_data = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train fused Hyperkähler model')
    parser.add_argument('--ae', type=str, default='convolutional', 
                        choices=['linear', 'convolutional'],
                        help='Autoencoder type')
    parser.add_argument('--transformer', type=str, default='stack',
                        choices=['attention', 'stack'],
                        help='Transformer type')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Latent dimension')
    parser.add_argument('--num-transformer-layers', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--save', type=str, default='model_checkpoint.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    logger.info(f"Configuration: ae={args.ae}, transformer={args.transformer}, "
                f"latent_dim={args.latent_dim}, epochs={args.epochs}")
    
    # Create config
    config = HyperkahlerConfig(
        ae_type=args.ae,
        transformer_type=args.transformer,
        latent_dim=args.latent_dim,
        num_transformer_layers=args.num_transformer_layers,
        fusion_enabled=True
    )
    
    # Create model
    model = ModelCombo(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load data
    logger.info("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size)
    logger.info(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, "
                f"Test: {len(test_loader)} batches")
    
    # Create trainer
    trainer = UnifiedTrainer(model, device=args.device, lr=args.lr)
    
    # Train
    logger.info(f"Starting training on {args.device}...")
    trainer.train(train_loader, val_loader, epochs=args.epochs)
    
    # Save checkpoint
    trainer.save_checkpoint(args.save)
    logger.info(f"Training complete. Model saved to {args.save}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.eval_epoch(test_loader)
    logger.info(f"Test loss: {test_metrics['val_loss']:.4f}")
    
    # Print history
    print("\n=== Training History ===")
    for epoch, (loss, recon, reg) in enumerate(zip(
        trainer.history["loss"],
        trainer.history["recon_loss"],
        trainer.history["hk_regularization"]
    )):
        print(f"Epoch {epoch+1}: loss={loss:.4f}, recon={recon:.4f}, reg={reg:.4f}")


if __name__ == '__main__':
    main()
