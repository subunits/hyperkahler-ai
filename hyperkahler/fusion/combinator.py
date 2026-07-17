"""Configuration-driven model combination from Sinusoidal-Hyperkahler-Fusion-Module."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from ..autoencoders import QuaternionLinearAE, HyperkahlerConvAE
from ..transformers import HyperKahlerAttention, HyperkahlerTransformer
from ..core import QuaternionOps


@dataclass
class HyperkahlerConfig:
    """Configuration for unified model."""
    ae_type: str = "linear"  # "linear" or "convolutional"
    transformer_type: str = "attention"  # "attention" or "stack"
    fusion_enabled: bool = True
    latent_dim: int = 64
    num_transformer_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    img_size: int = 28


class SinusoidalQuaternionEmbedding(nn.Module):
    """Sinusoidal embeddings for quaternions from fusion module."""
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Apply sinusoidal modulation to quaternion tensor.
        x: (batch, seq_len, dim) or (batch, dim)
        t: optional time/step values
        """
        if t is None:
            # Use sequence position as time
            seq_len = x.size(1) if x.dim() == 3 else 1
            t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        
        # Compute sinusoidal embeddings
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # (seq_len, dim)
        
        # Apply to quaternion (element-wise modulation)
        if x.dim() == 3:
            emb = emb.unsqueeze(0)  # (1, seq_len, dim)
        
        return x * emb


class ModelCombo(nn.Module):
    """Unified model combining any AE + any Transformer + optional Fusion."""
    
    def __init__(self, config: HyperkahlerConfig):
        super().__init__()
        self.config = config
        
        # Build autoencoder
        if config.ae_type == "linear":
            self.ae = QuaternionLinearAE(
                input_dim=config.img_size * config.img_size,
                latent_dim=config.latent_dim
            )
        elif config.ae_type == "convolutional":
            self.ae = HyperkahlerConvAE(
                latent_dim=config.latent_dim,
                img_size=config.img_size
            )
        else:
            raise ValueError(f"Unknown ae_type: {config.ae_type}")
        
        # Build transformer
        if config.transformer_type == "attention":
            self.transformer = HyperKahlerAttention(
                embed_dim=config.latent_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        elif config.transformer_type == "stack":
            self.transformer = HyperkahlerTransformer(
                embed_dim=config.latent_dim,
                num_layers=config.num_transformer_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            raise ValueError(f"Unknown transformer_type: {config.transformer_type}")
        
        # Optional fusion module
        if config.fusion_enabled:
            self.fusion = SinusoidalQuaternionEmbedding(config.latent_dim)
        else:
            self.fusion = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: encode → (fuse) → transform → decode
        """
        # Encode to latent
        z = self.ae.encode(x)
        
        # Reshape to sequence if needed
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (batch, 1, latent_dim)
        
        # Apply fusion modulation
        if self.fusion is not None:
            z = self.fusion(z)
        
        # Apply transformer
        z = self.transformer(z)
        
        # Decode back
        recon = self.ae.decode(z)
        
        return recon
    
    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (with fusion and transformer applied)."""
        z = self.ae.encode(x)
        
        if z.dim() == 2:
            z = z.unsqueeze(1)
        
        if self.fusion is not None:
            z = self.fusion(z)
        
        z = self.transformer(z)
        
        return z
    
    def compute_loss(self, x: torch.Tensor) -> dict:
        """Compute losses including Hyperkähler regularization."""
        recon = self.forward(x)
        
        # Reconstruction loss
        if isinstance(self.ae, HyperkahlerConvAE):
            recon_loss = nn.functional.mse_loss(recon, x)
        else:
            x_flat = x.view(x.size(0), -1)
            recon_loss = nn.functional.mse_loss(recon, x_flat)
        
        # Hyperkähler regularization
        z = self.ae.encode(x)
        if z.dim() == 2:
            z = z.unsqueeze(1)
        hk_reg = QuaternionOps.hyperkahler_regularizer(z) if z.shape[-1] == 4 else torch.tensor(0.0)
        
        total_loss = recon_loss + 0.01 * hk_reg
        
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "hk_regularization": hk_reg.item() if isinstance(hk_reg, torch.Tensor) else hk_reg
        }
