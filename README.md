# Hyperkähler AI — Unified Quaternion Framework

A consolidated PyTorch framework combining **two quaternion autoencoders** and **two quaternion transformers** into a single, flexible architecture for deep learning on Hyperkähler manifolds.

## Overview

This project unifies five specialized repositories into one coherent system:

- **Quaternion Linear Autoencoder** — Dense quaternion layers
- **Quaternion Convolutional Autoencoder** — Spatial feature extraction
- **Quaternion Attention Module** — Single-layer quaternion attention
- **Full Hyperkähler Transformer** — Multi-layer quaternion transformer stack
- **Sinusoidal Fusion Module** — Bridges any autoencoder to any transformer

All components operate on **quaternion-valued tensors** (4-component elements: r + i·I + j·J + k·K) and respect the **Hyperkähler geometry** of the data manifold.

## Architecture

```
Input
  ↓
[Autoencoder: Linear OR Convolutional]
  ↓
[Fusion: Sinusoidal Quaternion Embeddings]
  ↓
[Transformer: Attention Module OR Full Stack]
  ↓
[Shared Quaternion Core: Hamilton Product, Regularizers]
  ↓
Output
```

### 4 Possible Combinations

| AE Type | Transformer Type | Use Case |
|---------|------------------|----------|
| Linear | Attention | Fast prototyping, small latent |
| Linear | Stack | Sequence modeling on dense features |
| Convolutional | Attention | Image + local structure + quick attention |
| Convolutional | Stack | Image → deep hierarchical reasoning |

## Installation

```bash
# Clone and install
git clone https://github.com/subunits/hyperkahler-ai.git
cd hyperkahler-ai

# Option 1: Modern pyproject.toml (recommended)
pip install -e .

# Option 2: Setup.py (legacy)
pip install -e .[dev]

# Option 3: Direct requirements
pip install -r requirements.txt
```

## Quick Start

### Minimal Example

```python
from hyperkahler.fusion import ModelCombo, HyperkahlerConfig

# Create config
config = HyperkahlerConfig(
    ae_type="convolutional",
    transformer_type="stack",
    latent_dim=64,
    num_transformer_layers=2
)

# Instantiate model
model = ModelCombo(config)

# Forward pass
import torch
x = torch.randn(8, 1, 28, 28)  # Batch of MNIST-like images
output = model(x)  # (8, 1, 28, 28)

# Compute loss
losses = model.compute_loss(x)
print(f"Loss: {losses['loss'].item():.4f}")
```

### Training

```python
from hyperkahler.training import UnifiedTrainer
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO)

# Create trainer
trainer = UnifiedTrainer(model, device="cuda", lr=1e-3)

# Train on your dataset
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

trainer.train(train_loader, val_loader, epochs=10)

# Save checkpoint
trainer.save_checkpoint("model.pt")
```

### Using a Specific Combination

```python
# Linear AE + Attention (fast)
config = HyperkahlerConfig(ae_type="linear", transformer_type="attention")
model = ModelCombo(config)

# Conv AE + Full Transformer (heavy)
config = HyperkahlerConfig(ae_type="convolutional", transformer_type="stack", num_transformer_layers=4)
model = ModelCombo(config)
```

## Project Structure

```
hyperkahler-ai/
├── hyperkahler/
│   ├── core/              # Shared quaternion operations
│   │   ├── quaternion.py  # Hamilton product, norms, regularizers
│   │   └── geometry.py    # Hyperkähler metric, curvature
│   │
│   ├── autoencoders/      # Two AE variants
│   │   ├── linear.py      # QuaternionLinearAE
│   │   └── convolutional.py  # HyperkahlerConvAE
│   │
│   ├── transformers/      # Two transformer variants
│   │   ├── attention.py   # HyperKahlerAttention (single layer)
│   │   └── stack.py       # HyperkahlerTransformer (multi-layer)
│   │
│   ├── fusion/            # Flexible combinator
│   │   └── combinator.py  # ModelCombo + SinusoidalEmbedding
│   │
│   └── training/          # Unified trainer
│       └── trainer.py     # UnifiedTrainer
│
├── examples/              # Example scripts
├── tests/                 # Unit tests
├── pyproject.toml         # Modern Python packaging
├── setup.py               # Backward-compatible setup
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
└── LICENSE                # MIT License
```

## Key Components

### Quaternion Core

All operations are built on quaternions: **q = r + i·I + j·J + k·K**

```python
from hyperkahler.core import QuaternionOps

# Hamilton product
q1 = torch.randn(8, 64, 4)  # Batch of 64 quaternions
q2 = torch.randn(8, 64, 4)
product = QuaternionOps.hamilton_product(q1, q2)

# Regularization (enforce balanced imaginary components)
reg_loss = QuaternionOps.hyperkahler_regularizer(q1)
```

### Autoencoder Variants

```python
from hyperkahler.autoencoders import QuaternionLinearAE, HyperkahlerConvAE

# Dense layers
ae_linear = QuaternionLinearAE(input_dim=784, latent_dim=64)

# Convolutional (for images)
ae_conv = HyperkahlerConvAE(latent_dim=64, img_size=28)

# Encode/decode
z = ae_linear.encode(x)  # Get latent quaternions
recon = ae_linear.decode(z)
```

### Transformer Variants

```python
from hyperkahler.transformers import HyperKahlerAttention, HyperkahlerTransformer

# Single attention layer
attn = HyperKahlerAttention(embed_dim=64, num_heads=4)
output = attn(x)  # (batch, seq_len, 64)

# Full stack
transformer = HyperkahlerTransformer(embed_dim=64, num_layers=4)
output = transformer(x)
```

### Configuration

Control everything via `HyperkahlerConfig`:

```python
from hyperkahler.fusion import HyperkahlerConfig

config = HyperkahlerConfig(
    ae_type="convolutional",           # or "linear"
    transformer_type="stack",          # or "attention"
    latent_dim=64,
    num_transformer_layers=2,
    num_heads=4,
    dropout=0.1,
    img_size=28
)
```

## Training

Use `UnifiedTrainer` for any config:

```python
from hyperkahler.training import UnifiedTrainer

trainer = UnifiedTrainer(model, device="cuda", lr=1e-3)
metrics = trainer.train_epoch(train_loader)

# Access history
print(trainer.history["loss"])
```

Loss combines:
- **Reconstruction loss** (MSE)
- **Hyperkähler regularization** (balance imaginary components)

## Examples

See `examples/` directory:

- `train_linear_only.py` — Just linear autoencoder
- `train_conv_only.py` — Just convolutional autoencoder
- `train_attention_only.py` — Just attention module
- `train_full_transformer.py` — Full transformer
- `train_fused.py` — Full fusion pipeline

Run:
```bash
python examples/train_fused.py --ae convolutional --transformer stack --epochs 10
```

## Mathematics

### Quaternion Algebra

Hamilton product: **q₁ * q₂ = (r₁r₂ - v₁·v₂) + (r₁v₂ + r₂v₁ + v₁×v₂)**

Hyperkähler regularizer: Enforce **||i||² ≈ ||j||² ≈ ||k||²**

### Attention

Multi-head quaternion attention: **Attention(Q, K, V) = softmax(QK^T / √d)V**

Each head operates independently on quaternion projections.

## References

- **Quaternion Algebras**: Hamilton, 1843
- **Hyperkähler Geometry**: Beauville, 1985
- **Quaternion Neural Networks**: Parcollet et al., 2018
- **Transformer Architecture**: Vaswani et al., 2017

## Citation

```bibtex
@software{hyperkahler-ai,
  author = {Subunits},
  title = {Hyperkähler AI: Unified Quaternion Framework},
  year = {2024},
  url = {https://github.com/subunits/hyperkahler-ai}
}
```

## License

MIT License — see LICENSE file for details.

## Contributing

Pull requests welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new code
4. Submit a PR with description

## Questions?

Open an issue on GitHub or contact the maintainers.

---

**Built by [Subunits](https://subunits.dev)** — Quaternion AI at scale.
