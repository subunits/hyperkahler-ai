# Hyperkähler AI — Ready-to-Use Repository

## What's Inside

This ZIP contains a complete, unified GitHub repository structure combining all 5 of your repos:

```
hyperkahler-ai/
├── hyperkahler/              # Main package
│   ├── core/                 # Shared quaternion operations
│   ├── autoencoders/         # Linear + Convolutional variants
│   ├── transformers/         # Attention + Full Stack variants
│   ├── fusion/               # Flexible model combinator
│   └── training/             # Unified trainer
├── examples/                 # Training scripts
├── tests/                    # Unit tests
├── pyproject.toml            # Modern Python packaging
├── setup.py                  # Legacy setup script
├── requirements.txt          # Pinned dependencies
├── README.md                 # Full documentation
├── ARCHITECTURE.md           # Design deep-dive
├── LICENSE                   # MIT license
└── .gitignore               # Git ignore patterns
```

## Setup Instructions (iPad)

### Option 1: Upload to GitHub (Easiest)

1. **On GitHub.com** (in Safari):
   - Click **"+"** → **"New repository"**
   - Name: `hyperkahler-ai`
   - Description: "Unified Hyperkähler AI framework"
   - Uncheck "Add a README" (you have one)
   - Click **Create repository**

2. **Back on iPad**:
   - Download this ZIP to iPad
   - Open Files app → Downloads
   - Tap the ZIP → **"Extract"** (creates `hyperkahler-ai-repo/` folder)
   - Rename folder to `hyperkahler-ai` (optional)

3. **Using Working Copy (Git client for iPad)**:
   - Download **Working Copy** from App Store
   - Open Working Copy
   - Tap **"+"** → **"Create Repository"**
   - Pick the extracted `hyperkahler-ai` folder
   - Configure Git (name, email)
   - Add GitHub remote: `https://github.com/YOUR_USERNAME/hyperkahler-ai.git`
   - Commit all files with message: "Initial unified consolidation"
   - Push to GitHub

### Option 2: Manual GitHub Upload

1. Extract the ZIP on iPad (Files app)
2. Go to GitHub.com in Safari → your new repo
3. Click **"Add file"** → **"Upload files"**
4. Drag/select files from the extracted folder
5. Commit them one by one (tedious, but works)

### Option 3: Use a Mac/PC

If you have access to a computer:

```bash
# On Mac/Linux/Windows
unzip hyperkahler-ai.zip
cd hyperkahler-ai-repo
git init
git add .
git commit -m "Initial unified consolidation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/hyperkahler-ai.git
git push -u origin main
```

## Quick Start (After Upload to GitHub)

### Clone on iPad using Working Copy

1. Open Working Copy
2. Tap **"+"** → **"Clone Repository"**
3. Paste: `https://github.com/YOUR_USERNAME/hyperkahler-ai.git`
4. Tap **"Clone"**

### Install on Mac/Linux/Windows

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/hyperkahler-ai.git
cd hyperkahler-ai

# Install dependencies
pip install -r requirements.txt

# Or with dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Train a model
python examples/train_fused.py --ae convolutional --transformer stack --epochs 5
```

## File Structure at a Glance

| File/Folder | Purpose |
|------------|---------|
| `hyperkahler/core/` | Quaternion algebra, geometry |
| `hyperkahler/autoencoders/` | Linear & Convolutional AE |
| `hyperkahler/transformers/` | Attention & Full Transformer |
| `hyperkahler/fusion/` | Model combinator (any AE + Transformer) |
| `hyperkahler/training/` | Unified trainer & utilities |
| `examples/train_fused.py` | Example training script |
| `tests/test_quaternion.py` | Unit tests |
| `README.md` | User documentation |
| `ARCHITECTURE.md` | Technical deep-dive |
| `pyproject.toml` | Package metadata (modern) |
| `setup.py` | Legacy Python packaging |
| `requirements.txt` | Pinned dependency versions |

## Key Features

✅ **4 Model Combinations:**
- Linear AE + Attention Layer
- Linear AE + Full Transformer
- Convolutional AE + Attention Layer
- Convolutional AE + Full Transformer

✅ **Flexible Configuration:**
```python
from hyperkahler.fusion import ModelCombo, HyperkahlerConfig

config = HyperkahlerConfig(
    ae_type="convolutional",
    transformer_type="stack",
    latent_dim=64,
    num_transformer_layers=2
)
model = ModelCombo(config)
```

✅ **Unified Training:**
```python
from hyperkahler.training import UnifiedTrainer

trainer = UnifiedTrainer(model, device="cuda")
trainer.train(train_loader, val_loader, epochs=10)
```

✅ **All 5 Repos Consolidated:**
- No duplicate code
- Shared quaternion core
- Modular components
- Configuration-driven assembly

## Next Steps

1. **Extract the ZIP** on iPad
2. **Upload to GitHub** (using Working Copy or manual upload)
3. **Clone locally** (on Mac/PC or back to iPad)
4. **Install:** `pip install -r requirements.txt`
5. **Run example:** `python examples/train_fused.py`
6. **Experiment:** Try different `--ae` and `--transformer` options

## Questions?

- Check **README.md** for usage examples
- Check **ARCHITECTURE.md** for technical details
- Read docstrings in the code (well-commented)

---

**Ready to use!** 🚀 This repository is fully functional and ready for training, experimentation, and extension.
