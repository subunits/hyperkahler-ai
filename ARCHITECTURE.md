# Architecture Documentation

## Overview

The unified Hyperkähler AI framework consolidates five specialized repositories into a single, modular design. All components operate on quaternion-valued tensors and respect Hyperkähler manifold geometry.

## Core Principles

1. **Quaternion Algebra** — All operations use Hamilton products and quaternion norms
2. **Modularity** — Each component (AE, Transformer) is independently useful
3. **Flexibility** — Config-driven model assembly (any AE + any Transformer)
4. **Geometric Regularization** — Enforces balanced imaginary components (I, J, K)

## Module Breakdown

### 1. Core (`hyperkahler/core/`)

**quaternion.py**
- `QuaternionOps.hamilton_product()` — Quaternion multiplication
- `QuaternionOps.quaternion_norm()` — Magnitude of quaternions
- `QuaternionOps.quaternion_conjugate()` — Conjugate (negate imaginary parts)
- `QuaternionOps.hyperkahler_regularizer()` — Regularizes imaginary balance

**geometry.py**
- `GeometryOps.project_to_complex_structures()` — Decompose q into I, J, K
- `GeometryOps.kahler_metric()` — Local Kähler metric norm
- `GeometryOps.curvature_penalty()` — Penalty for deviation from Hyperkähler structure

### 2. Autoencoders (`hyperkahler/autoencoders/`)

#### QuaternionLinearAE (from Hyperkahler-AI-Quaternion-Autoencoder)

```
Input (batch, dim) 
  → Linear projection (dim → latent_dim)
  → Reshape to quaternions (batch, latent_dim/4, 4)
  → [Latent Space]
  → Linear projection (latent_dim → dim)
  → Output (batch, dim)
```

**Use case:** Fully connected, fast, good for dense features

#### HyperkahlerConvAE (from Hyperkahler-Convolutional-Autoencoder)

```
Input (batch, 1, 28, 28)
  → Conv 3x3 (1 → 16, stride 2)
  → Conv 3x3 (16 → 32, stride 2)
  → Flatten + Linear → latent_dim
  → [Latent Space]
  → Linear → 32 * (H/4) * (W/4)
  → ConvTranspose 3x3 (32 → 16, stride 2)
  → ConvTranspose 3x3 (16 → 1, stride 2)
  → Output (batch, 1, 28, 28)
```

**Use case:** Images/spatial data, preserves local structure

**Key difference:** Conv AE reconstructs spatial structure; Linear AE treats input as flat vector.

### 3. Transformers (`hyperkahler/transformers/`)

#### HyperKahlerAttention (from HyperKahler-Transformer-Module)

Single multi-head attention layer:

```
Input (batch, seq_len, embed_dim)
  → Q/K/V projections
  → Split into num_heads
  → Attention(Q, K, V) = softmax(QK^T / √d)V
  → Concatenate heads
  → Output projection
  → Output (batch, seq_len, embed_dim)
```

**Use case:** Quick attention over sequences, good for fusion module output

**Regularization:** Optional hyperkähler_regularization() enforces quaternion balance

#### HyperkahlerTransformer (from Hyperkahler-Transformer)

Multi-layer stack with residual connections:

```
Input (batch, seq_len, embed_dim)
  → Positional embeddings (learned)
  → For each of num_layers layers:
      → MultiHeadAttention + Residual + LayerNorm
      → FFN (linear → ReLU → linear) + Residual + LayerNorm
  → Output (batch, seq_len, embed_dim)
```

**Use case:** Deep reasoning over sequences, hierarchical feature extraction

**Key difference:** Attention is single-layer; Transformer stacks N blocks.

### 4. Fusion (`hyperkahler/fusion/`)

#### SinusoidalQuaternionEmbedding (from Sinusoidal-Hyperkahler-Fusion-Module)

Modulates quaternion features with sinusoidal patterns:

```
Input quaternions z (batch, seq_len, dim)
  → Compute sinusoidal frequencies (fixed or learnable)
  → Element-wise multiplication: z_modulated = z ⊙ sin(frequencies * t)
  → Output (batch, seq_len, dim)
```

**Purpose:** 
- Adds positional/temporal information without learned parameters
- Preserves quaternion structure
- Bridges discrete latent space to continuous time domain

#### ModelCombo (Config-Driven Assembly)

Orchestrates the full pipeline:

```
HyperkahlerConfig:
  {
    "ae_type": "convolutional" | "linear",
    "transformer_type": "attention" | "stack",
    "fusion_enabled": True | False,
    "latent_dim": 64,
    ...
  }
  ↓
ModelCombo.__init__():
  → Instantiate AE (linear or conv)
  → Instantiate Transformer (attention or stack)
  → Optionally instantiate Fusion
  ↓
ModelCombo.forward(x):
  1. z = ae.encode(x)           # Compress to latent quaternions
  2. if fusion: z = fusion(z)   # Modulate with sinusoidal
  3. z = transformer(z)         # Apply global attention
  4. out = ae.decode(z)         # Reconstruct
```

**Loss computation:**
```
total_loss = reconstruction_loss + 0.01 * hyperkahler_regularizer_loss
```

### 5. Training (`hyperkahler/training/`)

#### UnifiedTrainer

Handles training for any `ModelCombo`:

```python
trainer = UnifiedTrainer(model, device="cuda", lr=1e-3)
trainer.train(train_loader, val_loader, epochs=10)
trainer.save_checkpoint("model.pt")
```

**Features:**
- Automatic loss computation (recon + regularization)
- Training history tracking
- Checkpoint save/load
- Gradient updates

## Data Flow Diagram

```
                    ModelCombo Configuration
                           │
                ┌──────────┼──────────┐
                │          │          │
            [Linear AE]  [Conv AE]   │
                │          │          │
                └──────────┴──────────┘
                           │
                     [encode(x)]
                           │
                 (batch, latent_dim/4, 4)
                           │
                    [Fusion Module]
                  (Sinusoidal Modulation)
                           │
                 (batch, latent_dim/4, 4)
                           │
                  ┌────────┴────────┐
              [Attn]           [Stack]
                │                 │
                └────────┬────────┘
                         │
                    [decode(x)]
                         │
                   [Reconstruction]
                         │
                    [Loss Backprop]
```

## 4 Main Combinations

### 1. Linear + Attention (Lightweight)
- **Encoding:** Dense layers → flatten to quaternions
- **Fusion:** Sinusoidal modulation
- **Attention:** Single fast attention layer
- **Use:** Quick experiments, small data, inference latency-critical

### 2. Linear + Stack (Sequence Modeling)
- **Encoding:** Dense → quaternions
- **Fusion:** Sinusoidal
- **Transformer:** 4-layer stack with multi-head attention
- **Use:** Structured sequential data, moderate compute

### 3. Conv + Attention (Image + Quick)
- **Encoding:** Convolution → quaternions (preserves spatial structure)
- **Fusion:** Sinusoidal
- **Attention:** Single fast layer
- **Use:** Images/spatial + fast attention, medium compute

### 4. Conv + Stack (Image + Deep)
- **Encoding:** Convolution → quaternions
- **Fusion:** Sinusoidal
- **Transformer:** 4-layer stack
- **Use:** Complex image tasks, maximum expressivity, highest compute

## Quaternion Representation

All intermediate tensors have shape `(..., 4)` representing `[r, i, j, k]`:

```python
# Quaternion (batch, seq_len, 4)
q = torch.randn(32, 100, 4)  # 32 samples, 100-step sequences, 4-component quaternions

# Hamilton product (example)
q1 = torch.randn(32, 4)
q2 = torch.randn(32, 4)
q3 = QuaternionOps.hamilton_product(q1, q2)  # (32, 4)

# Regularization ensures ||i||² ≈ ||j||² ≈ ||k||²
reg = QuaternionOps.hyperkahler_regularizer(q)  # scalar loss term
```

## Loss Function

```
L_total = L_recon + λ * L_reg

L_recon = MSE(output, input)
L_reg = ||norm(i) - mean_norm||² + ||norm(j) - mean_norm||² + ||norm(k) - mean_norm||²
λ = 0.01  (tunable)
```

The regularization term enforces **Hyperkähler structure** by keeping the three complex structures (I, J, K) balanced.

## Extension Points

### Adding a New Autoencoder
1. Inherit from `nn.Module`
2. Implement `encode(x) → z` and `decode(z) → x`
3. Register in `hyperkahler/autoencoders/__init__.py`
4. Update `ModelCombo._build_ae()` to instantiate it
5. Set `config.ae_type = "my_custom_ae"`

### Adding a New Transformer
1. Inherit from `nn.Module`
2. Implement `forward(x) → output`
3. Register in `hyperkahler/transformers/__init__.py`
4. Update `ModelCombo._build_transformer()` to instantiate it
5. Set `config.transformer_type = "my_custom_transformer"`

### Custom Loss Function
1. Subclass `UnifiedTrainer`
2. Override `compute_loss()` method in the model or trainer
3. Add your regularizers/penalties

## Performance Considerations

| Config | Params | Inference | Training | Use Case |
|--------|--------|-----------|----------|----------|
| Linear + Attn | ~200K | <1ms | Fast | Prototyping |
| Linear + Stack | ~800K | 2ms | Moderate | Baseline |
| Conv + Attn | ~500K | 5ms | Moderate | Image + speed |
| Conv + Stack | ~2M | 20ms | Slow | Full power |

(Rough estimates on MNIST with 64 latent dims)

## References

1. **Original 5 Repos:**
   - Hyperkahler-AI-Quaternion-Autoencoder
   - Hyperkahler-Convolutional-Autoencoder
   - HyperKahler-Transformer-Module
   - Hyperkahler-Transformer
   - Sinusoidal-Hyperkahler-Fusion-Module

2. **Quaternion Theory:**
   - Hamilton, W. R. (1843). "On a New Species of Imaginary Quantities"
   - Quaternion Algebras (https://en.wikipedia.org/wiki/Quaternion)

3. **Quaternion Neural Networks:**
   - Parcollet, T., et al. (2018). "Quaternion Neural Networks"
   - Zhu, X., et al. (2018). "Quaternion Recurrent Neural Networks"

4. **Hyperkähler Geometry:**
   - Beauville, A. (1985). "Variétés Kähleriennes dont la première classe de Chern est nulle"
   - Salamon, S. M. (1989). "Quaternionic Kähler Manifolds"

5. **Transformers:**
   - Vaswani, A., et al. (2017). "Attention Is All You Need"
