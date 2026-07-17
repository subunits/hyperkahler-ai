"""Unit tests for quaternion operations."""

import torch
import pytest
from hyperkahler.core import QuaternionOps


def test_hamilton_product():
    """Test Hamilton product of two quaternions."""
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Identity
    q2 = torch.tensor([[2.0, 3.0, 4.0, 5.0]])  # Some quaternion
    
    result = QuaternionOps.hamilton_product(q1, q2)
    
    # Identity * q2 should equal q2
    assert torch.allclose(result, q2)


def test_quaternion_norm():
    """Test quaternion norm."""
    q = torch.tensor([[3.0, 4.0, 0.0, 0.0]])  # 3-4-0-0 vector
    
    norm = QuaternionOps.quaternion_norm(q)
    expected = torch.tensor([[5.0]])  # sqrt(9 + 16 + 0 + 0) = 5
    
    assert torch.allclose(norm, expected)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    conj = QuaternionOps.quaternion_conjugate(q)
    expected = torch.tensor([[1.0, -2.0, -3.0, -4.0]])
    
    assert torch.allclose(conj, expected)


def test_hyperkahler_regularizer():
    """Test Hyperkähler regularizer."""
    # Balanced quaternions (should have low regularization)
    q_balanced = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])  # All equal
    reg_balanced = QuaternionOps.hyperkahler_regularizer(q_balanced, weight=1.0)
    
    # Unbalanced quaternions (should have high regularization)
    q_unbalanced = torch.tensor([[[1.0, 10.0, 0.1, 0.1]]])  # i dominates
    reg_unbalanced = QuaternionOps.hyperkahler_regularizer(q_unbalanced, weight=1.0)
    
    # Unbalanced should have higher regularization
    assert reg_unbalanced > reg_balanced


def test_quaternion_batch():
    """Test operations on batches of quaternions."""
    batch_size = 16
    q1 = torch.randn(batch_size, 4)
    q2 = torch.randn(batch_size, 4)
    
    # Hamilton product on batch
    result = QuaternionOps.hamilton_product(q1, q2)
    assert result.shape == (batch_size, 4)
    
    # Norms
    norms = QuaternionOps.quaternion_norm(q1)
    assert norms.shape == (batch_size, 1)


if __name__ == "__main__":
    test_hamilton_product()
    test_quaternion_norm()
    test_quaternion_conjugate()
    test_hyperkahler_regularizer()
    test_quaternion_batch()
    print("All tests passed!")
