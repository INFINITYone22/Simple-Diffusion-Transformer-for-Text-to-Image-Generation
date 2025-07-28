# Simple Diffusion Transformer

A lightweight, FP8-optimized diffusion transformer for text-to-image generation. Pure transformer architecture trained from scratch without pre-trained encoders.

## Features

- **Pure Transformer**: End-to-end learnable model without VAE bottlenecks
- **FP8 Precision**: Maximum performance with E4M3/E5M2 formats
- **High Resolution**: Native 1024×1024 image generation
- **Scalable**: 8B, 12B, and 16B parameter configurations
- **Unlimited Prompts**: Dynamic text handling with RoPE

## Quick Start

**Install:**
```bash
pip install torch torchvision transformer-engine flash-attn pillow tqdm
```

**Train:**
```bash
python train.py --config 8b --batch_size 16
```

**Generate:**
```bash
python generate.py --prompt "a fluffy cat on a windowsill" --model 8b --checkpoint model.pt
```

## Model Configs

| Size | Parameters | Dimension | Layers |
|------|------------|-----------|--------|
| 8B   | ~8.1B     | 4096      | 32     |
| 12B  | ~12.2B    | 5120      | 32     |
| 16B  | ~16.3B    | 6144      | 36     |

## Files

- `model.py` - Core transformer implementation
- `train.py` - Training script with FP8 support
- `generate.py` - Image generation from text prompts
