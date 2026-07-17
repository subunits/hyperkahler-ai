"""Linear Quaternion Autoencoder from Hyperkahler-AI-Quaternion-Autoencoder."""

import torch
import torch.nn as nn
from ..core import QuaternionOps


class QuaternionLinearLayer(nn.Module):
    """Quaternion linear layer: y = q * x (Hamilton product)."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix (quaternion weights)
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 4) / (in_features ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 4))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Hamilton product.
        x: (batch, in_features, 4)
        output: (batch, out_features, 4)
        """
        batch_size = x.size(0)
        
        # Compute Hamilton products
        output = torch.zeros(batch_size, self.out_features, 4, device=x.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                product = QuaternionOps.hamilton_product(
                    x[:, j, :].unsqueeze(1),  # (batch, 1, 4)
                    self.weight[i, j, :].unsqueeze(0).expand(batch_size, -1)  # (batch, 4)
                )
                output[:, i, :] += product.squeeze(1)
        
        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        
        return output


class QuaternionLinearAE(nn.Module):
    """Linear Quaternion Autoencoder."""
    
    def __init__(self, input_dim: int = 28 * 28, latent_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: flatten input, project to quaternions
        self.encoder_linear = nn.Linear(input_dim, latent_dim)
        
        # Decoder: project back from latent to input
        self.decoder_linear = nn.Linear(latent_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent quaternion space."""
        # Flatten
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Project to latent
        z = self.encoder_linear(x_flat)
        
        # Reshape to quaternions (latent_dim must be divisible by 4)
        if self.latent_dim % 4 == 0:
            z = z.view(batch_size, self.latent_dim // 4, 4)
        else:
            z = z.view(batch_size, self.latent_dim)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent to reconstruction."""
        batch_size = z.size(0)
        
        # Flatten if quaternion
        if z.dim() == 3:
            z_flat = z.view(batch_size, -1)
        else:
            z_flat = z
        
        # Project to output
        recon = self.decoder_linear(z_flat)
        
        return recon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
