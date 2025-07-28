Simple Diffusion Transformer for Text-to-Image Generation
Features
From-Scratch Training: Simple embeddings for text and images, concatenated for cross-modal conditioning.

Diffusion Process: Iterative denoising guided by self-attention in a transformer backbone.

FP8 Precision: Models are optimized for FP8 (E4M3 forward, E5M2 backward) to reduce memory and improve speed on compatible hardware (e.g., NVIDIA H100/A100 GPUs).

Dynamic Prompts: Handles arbitrary-length text inputs with Rotary Positional Embeddings (RoPE).

Scalable Configurations: Three model sizes (~8B, ~12B, ~16B parameters) for different compute needs.

Model Configurations
Config	Hidden Dim	Layers	Attention Heads	Estimated Params
8B Model	4096	32	32	~8.1B
12B Model	5120	32	40	~12.2B
16B Model	6144	36	48	~16.3B
These configs balance depth for detailed prompt handling and width for feature richness, all in FP8 precision.
BY ROHITH GARAPATI 
