"""Unified training loop for Hyperkähler models."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List
import logging


logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """Trainer for any ModelCombo configuration."""
    
    def __init__(self, model, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.history = {
            "loss": [],
            "recon_loss": [],
            "hk_regularization": []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_reg = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            losses = self.model.compute_loss(x)
            loss = losses["loss"]
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_recon += losses["recon_loss"].item()
            total_reg += losses["hk_regularization"]
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: loss={loss.item():.4f}")
        
        # Average over batches
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_reg = total_reg / num_batches
        
        self.history["loss"].append(avg_loss)
        self.history["recon_loss"].append(avg_recon)
        self.history["hk_regularization"].append(avg_reg)
        
        return {
            "loss": avg_loss,
            "recon_loss": avg_recon,
            "hk_regularization": avg_reg
        }
    
    def eval_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                
                losses = self.model.compute_loss(x)
                total_loss += losses["loss"].item()
                num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 10):
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}: {train_metrics}")
            
            if val_loader is not None:
                val_metrics = self.eval_epoch(val_loader)
                logger.info(f"Validation: {val_metrics}")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = checkpoint["history"]
        logger.info(f"Checkpoint loaded from {path}")
