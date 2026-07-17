"""Full Hyperkähler Transformer from Hyperkahler-Transformer."""

import torch
import torch.nn as nn
from .attention import HyperKahlerAttention


class TransformerBlock(nn.Module):
    """Single transformer block: attention + feedforward."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attention = HyperKahlerAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual attention + feedforward."""
        # Self-attention with residual
        attn_out = self.attention(x)
        x = x + attn_out
        x = self.norm1(x)
        
        # Feedforward with residual
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)
        
        return x


class HyperkahlerTransformer(nn.Module):
    """Multi-layer Hyperkähler Transformer."""
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 512,
        max_seq_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        output: (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        
        # Add positional embeddings
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_embed = self.pos_embedding(pos)
        x = x + pos_embed
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence through transformer."""
        return self.forward(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """For autoencoder compatibility, identity function."""
        return x
