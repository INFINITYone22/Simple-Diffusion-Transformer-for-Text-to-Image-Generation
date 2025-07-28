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

## Requirements

- GPU with 24GB+ VRAM
- CUDA 11.8+ for FP8 support
- PyTorch 2.1+

## License

MIT License

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3696807/b1509ea7-aeeb-450e-aba5-700162fe2bb7/PAPER.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/3696807/f608b8c4-bd29-47db-a7dd-6a1136062c83/Paper-Analysis-and-Report-Generation.pdf
