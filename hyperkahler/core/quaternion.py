"""Quaternion algebra and operations shared across all modules."""

import torch
import torch.nn as nn


class QuaternionOps:
    """Static methods for quaternion operations."""
    
    @staticmethod
    def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamilton product of two quaternions.
        q1, q2 shape: (..., 4) where components are [r, i, j, k]
        """
        r1, i1, j1, k1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        r2, i2, j2, k2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
        i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
        j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
        k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2
        
        return torch.stack([r, i, j, k], dim=-1)
    
    @staticmethod
    def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
        """Compute norm of quaternions."""
        return torch.norm(q, dim=-1, keepdim=True)
    
    @staticmethod
    def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Return conjugate of quaternion (negate imaginary parts)."""
        conj = q.clone()
        conj[..., 1:] = -conj[..., 1:]
        return conj
    
    @staticmethod
    def hyperkahler_regularizer(tensor: torch.Tensor, weight: float = 0.1) -> torch.Tensor:
        """
        Regularizer enforcing balanced imaginary components (I, J, K structures).
        Encourages ||i||² ≈ ||j||² ≈ ||k||²
        """
        if tensor.shape[-1] < 4:
            return torch.tensor(0.0, device=tensor.device)
        
        i_norm = tensor[..., 1].norm()
        j_norm = tensor[..., 2].norm()
        k_norm = tensor[..., 3].norm()
        
        mean_norm = (i_norm + j_norm + k_norm) / 3.0
        reg = weight * ((i_norm - mean_norm).pow(2) + 
                        (j_norm - mean_norm).pow(2) + 
                        (k_norm - mean_norm).pow(2))
        return reg
