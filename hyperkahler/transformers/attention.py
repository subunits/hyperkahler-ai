"""Quaternion Attention Module from HyperKahler-Transformer-Module."""

import torch
import torch.nn as nn
import math
from ..core import QuaternionOps


class HyperKahlerAttention(nn.Module):
    """
    Quaternion-aware self-attention mechanism.
    Applies attention over quaternion-valued sequences.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V projections (real-valued but will operate on quaternions)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, embed_dim)
        output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output
    
    def hyperkahler_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hyperkähler regularization to attention output."""
        if x.shape[-1] % 4 != 0:
            return torch.tensor(0.0, device=x.device)
        
        # Reshape as quaternions
        x_quat = x.view(*x.shape[:-1], -1, 4)
        
        # Apply regularizer
        reg = QuaternionOps.hyperkahler_regularizer(x_quat)
        return reg
