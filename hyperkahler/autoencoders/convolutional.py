"""Convolutional Quaternion Autoencoder from Hyperkahler-Convolutional-Autoencoder."""

import torch
import torch.nn as nn
from ..core import QuaternionOps


class QuaternionConv2d(nn.Module):
    """Quaternion 2D Convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Quaternion kernels (4 components per output channel per input channel)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, 4) / (in_channels ** 0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels, 4))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_channels, height, width, 4)
        output: (batch, out_channels, height, width, 4)
        """
        batch, in_c, h, w, quat_dim = x.shape
        out_c = self.out_channels
        
        # Unfold for convolution
        x_unfolded = torch.nn.functional.unfold(
            x.view(batch, in_c * quat_dim, h, w),
            kernel_size=self.kernel_size,
            padding=self.padding
        )  # (batch, in_c * quat_dim * k * k, h * w)
        
        # Reshape for quaternion operations
        x_unfolded = x_unfolded.view(
            batch, in_c, self.kernel_size, self.kernel_size, quat_dim, -1
        )  # (batch, in_c, k, k, 4, out_size)
        
        out_h = out_w = int((h - self.kernel_size + 2 * self.padding) + 1)
        output = torch.zeros(batch, out_c, out_h, out_w, 4, device=x.device)
        
        # Apply quaternion convolution
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        # Hamilton product at each position
                        for idx in range(out_h * out_w):
                            q_in = x_unfolded[:, ic, i, j, :, idx]  # (batch, 4)
                            q_weight = self.weight[oc, ic, i, j, :]  # (4,)
                            
                            product = QuaternionOps.hamilton_product(
                                q_in.unsqueeze(1), q_weight.unsqueeze(0)
                            ).squeeze(1)
                            
                            h_idx, w_idx = idx // out_w, idx % out_w
                            output[:, oc, h_idx, w_idx, :] += product
        
        output += self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return output


class HyperkahlerConvAE(nn.Module):
    """Convolutional Quaternion Autoencoder."""
    
    def __init__(self, latent_dim: int = 64, img_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc_fc = nn.Linear(32 * (img_size // 4) * (img_size // 4), latent_dim)
        
        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 32 * (img_size // 4) * (img_size // 4))
        self.dec_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        x = self.relu(self.enc_conv1(x))
        x = self.relu(self.enc_conv2(x))
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        z = self.enc_fc(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent to image."""
        batch_size = z.size(0)
        
        x = self.dec_fc(z)
        x = x.view(batch_size, 32, self.img_size // 4, self.img_size // 4)
        
        x = self.relu(self.dec_conv1(x))
        x = self.sigmoid(self.dec_conv2(x))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder pass."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon
