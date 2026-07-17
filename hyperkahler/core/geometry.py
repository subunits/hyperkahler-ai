"""Hyperkähler geometry utilities (I, J, K complex structures)."""

import torch
import torch.nn as nn


class GeometryOps:
    """Hyperkähler manifold structures."""
    
    @staticmethod
    def project_to_complex_structures(q: torch.Tensor) -> dict:
        """
        Project quaternion into three complex structures I, J, K.
        Returns dict with .I, .J, .K as complex tensors.
        """
        # q = r + i*I + j*J + k*K
        # Extract real and imaginary parts
        r = q[..., 0]  # Real part
        i = q[..., 1]  # I structure
        j = q[..., 2]  # J structure
        k = q[..., 3]  # K structure
        
        # Return as complex structures (for analysis/visualization)
        return {
            'I': torch.complex(r, i),  # Complex I structure
            'J': torch.complex(r, j),  # Complex J structure
            'K': torch.complex(r, k),  # Complex K structure
        }
    
    @staticmethod
    def kahler_metric(q: torch.Tensor) -> torch.Tensor:
        """
        Compute local Kähler metric norm.
        Measures "geodesic distance" in quaternionic space.
        """
        r = q[..., 0]
        i = q[..., 1]
        j = q[..., 2]
        k = q[..., 3]
        
        # Kähler metric: metric on quaternionic projective space
        metric = r.pow(2) + i.pow(2) + j.pow(2) + k.pow(2)
        return torch.sqrt(metric + 1e-8)
    
    @staticmethod
    def curvature_penalty(q: torch.Tensor) -> torch.Tensor:
        """
        Penalty for deviating from Hyperkähler structure.
        Lower values indicate data closer to a Hyperkähler manifold.
        """
        # Simplified: penalize if any complex structure dominates
        r = q[..., 0].abs()
        i = q[..., 1].abs()
        j = q[..., 2].abs()
        k = q[..., 3].abs()
        
        mean_mag = (r + i + j + k) / 4.0
        penalty = ((r - mean_mag).pow(2) + 
                   (i - mean_mag).pow(2) + 
                   (j - mean_mag).pow(2) + 
                   (k - mean_mag).pow(2)).mean()
        return penalty
