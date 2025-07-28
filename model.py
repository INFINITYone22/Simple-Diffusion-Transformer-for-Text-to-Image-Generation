import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

class RotaryPositionalEmbedding(nn.Module):
    """Optimized RoPE implementation for FP8"""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cached_cos = emb.cos().to(dtype)
            self._cached_sin = emb.sin().to(dtype)
            self._cached_seq_len = seq_len
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, 
                           seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._compute_cos_sin(seq_len, q.device, q.dtype)
        
        def rotate_half(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            return torch.cat((-x2, x1), dim=-1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed

class FP8TextEmbedding(nn.Module):
    """FP8-optimized text embedding with dynamic length support"""
    def __init__(self, vocab_size: int = 50000, dim: int = 4096, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Use transformer-engine for FP8 embedding
        self.token_embed = te.Linear(vocab_size, dim, bias=False)
        self.rope = RotaryPositionalEmbedding(dim // 32)  # Assuming 32 heads
        
        # Learned positional encoding as fallback
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        B, seq_len = text_tokens.shape
        
        # One-hot encoding for FP8 compatibility
        one_hot = F.one_hot(text_tokens, num_classes=self.vocab_size).float()
        embeds = self.token_embed(one_hot)
        
        # Add positional embeddings
        embeds = embeds + self.pos_embed[:, :seq_len, :]
        
        return embeds

class FP8ImageEmbedding(nn.Module):
    """FP8-optimized image patch embedding"""
    def __init__(self, patch_size: int = 16, dim: int = 4096, image_size: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # FP8 projection layer
        self.proj = te.Linear(3 * patch_size * patch_size, dim, bias=False)
        
        # 2D positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Learnable class token for global image context
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        
        # Extract patches: [B, num_patches, patch_size^2 * C]
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * self.patch_size * self.patch_size)
        
        # Project to embedding dimension
        patch_embeds = self.proj(patches)
        
        # Add positional embeddings
        patch_embeds = patch_embeds + self.pos_embed
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        return embeddings

class FP8AttentionBlock(nn.Module):
    """Transformer block optimized for FP8 with flash attention"""
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # FP8 attention layers
        self.norm1 = te.LayerNorm(dim)
        self.q_proj = te.Linear(dim, dim, bias=False)
        self.k_proj = te.Linear(dim, dim, bias=False)
        self.v_proj = te.Linear(dim, dim, bias=False)
        self.out_proj = te.Linear(dim, dim, bias=False)
        
        # FP8 MLP
        self.norm2 = te.LayerNorm(dim)
        self.mlp = nn.Sequential(
            te.Linear(dim, dim * mlp_ratio, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            te.Linear(dim * mlp_ratio, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        # AdaLN modulation for timestep conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            te.Linear(dim, 6 * dim, bias=True)
        )
        
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # AdaLN modulation
        modulation = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        
        # Pre-norm with modulation
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        # Multi-head attention with RoPE
        q = self.q_proj(x_norm).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to q and k
        q, k = self.rope.apply_rotary_pos_emb(q, k, N)
        
        # Flash attention (FP8 compatible)
        with te.fp8_autocast(enabled=True):
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection with gate
        x = x + gate_msa.unsqueeze(1) * attn_output
        
        # MLP with modulation
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm)
        
        # Residual connection with gate
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x

class FP8DiffusionTransformer(nn.Module):
    """Main diffusion transformer model with full FP8 support"""
    def __init__(self, config: str = '8b', image_size: int = 1024, patch_size: int = 16):
        super().__init__()
        
        # Model configurations
        configs = {
            '8b': {'dim': 4096, 'layers': 32, 'heads': 32, 'mlp_ratio': 4},
            '12b': {'dim': 5120, 'layers': 32, 'heads': 40, 'mlp_ratio': 4},
            '16b': {'dim': 6144, 'layers': 36, 'heads': 48, 'mlp_ratio': 4}
        }
        
        cfg = configs.get(config, configs['8b'])
        self.dim = cfg['dim']
        self.layers = cfg['layers']
        self.heads = cfg['heads']
        self.mlp_ratio = cfg['mlp_ratio']
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Embeddings
        self.text_embed = FP8TextEmbedding(dim=self.dim)
        self.image_embed = FP8ImageEmbedding(patch_size=patch_size, dim=self.dim, image_size=image_size)
        
        # Timestep embedding with Fourier features
        timestep_dim = 256
        self.timestep_mlp = nn.Sequential(
            nn.Linear(timestep_dim, self.dim),
            nn.SiLU(),
            te.Linear(self.dim, self.dim, bias=True)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FP8AttentionBlock(self.dim, self.heads, self.mlp_ratio)
            for _ in range(self.layers)
        ])
        
        # Output layers
        self.final_norm = te.LayerNorm(self.dim)
        self.final_linear = te.Linear(self.dim, 3 * patch_size * patch_size, bias=True)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, te.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, te.LayerNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def timestep_embedding(self, timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
        """Sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, noisy_images: torch.Tensor, text_tokens: torch.Tensor, 
                timesteps: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = noisy_images.shape[0]
        
        # Embeddings
        text_emb = self.text_embed(text_tokens)  # [B, text_len, dim]
        image_emb = self.image_embed(noisy_images)  # [B, num_patches + 1, dim]
        
        # Concatenate text and image embeddings
        x = torch.cat([text_emb, image_emb], dim=1)  # [B, text_len + num_patches + 1, dim]
        
        # Timestep embedding
        t_emb_raw = self.timestep_embedding(timesteps)
        t_emb = self.timestep_mlp(t_emb_raw)  # [B, dim]
        
        # Create attention mask
        text_len = text_tokens.shape[1]
        total_len = x.shape[1]
        attention_mask = torch.ones(B, total_len, total_len, device=x.device, dtype=torch.bool)
        
        if text_mask is not None:
            # Mask out padding tokens in text
            attention_mask[:, :text_len, :] = text_mask.unsqueeze(-1)
            attention_mask[:, :, :text_len] = text_mask.unsqueeze(-2)
        
        # Forward through transformer blocks with FP8
        with te.fp8_autocast(enabled=True):
            for block in self.blocks:
                x = block(x, t_emb, attention_mask)
        
        # Extract image tokens (skip class token)
        image_tokens = x[:, text_len + 1:]  # [B, num_patches, dim]
        
        # Predict noise for each patch
        x = self.final_norm(image_tokens)
        noise_pred = self.final_linear(x)  # [B, num_patches, 3 * patch_size^2]
        
        # Reshape to image format
        noise_pred = noise_pred.view(B, self.num_patches, 3, self.patch_size, self.patch_size)
        
        # Convert patches back to image
        patches_per_side = self.image_size // self.patch_size
        noise_pred = noise_pred.view(B, patches_per_side, patches_per_side, 3, self.patch_size, self.patch_size)
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5).contiguous()
        noise_pred = noise_pred.view(B, 3, self.image_size, self.image_size)
        
        return noise_pred

def create_model(config: str = '8b', **kwargs) -> FP8DiffusionTransformer:
    """Factory function to create model with FP8 recipe"""
    model = FP8DiffusionTransformer(config=config, **kwargs)
    
    # Set FP8 recipe for optimal performance
    fp8_recipe = recipe.DelayedScaling(
        fp8_format=recipe.Format.E4M3,  # Forward pass
        amax_history_len=1024,
        amax_compute_algo="max"
    )
    
    return model
