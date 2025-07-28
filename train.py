import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import numpy as np
from tqdm import tqdm
import transformer_engine.pytorch as te
from model import create_model

class DDPMScheduler:
    """DDPM noise scheduler optimized for FP8"""
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to the schedule"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

def setup_distributed():
    """Setup for distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def main(args):
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Create model with FP8 support
    model = create_model(config=args.config, image_size=args.image_size)
    model = model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Optimizer with FP8 support
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.learning_rate * 0.1
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(num_timesteps=1000)
    
    # Enable FP8 training
    fp8_recipe = te.recipe.DelayedScaling(
        fp8_format=te.recipe.Format.E4M3,
        amax_history_len=1024,
        amax_compute_algo="max"
    )
    
    # Training loop
    model.train()
    step = 0
    
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        while step < args.max_steps:
            # Simulate batch loading (replace with your dataset)
            batch_size = args.batch_size // world_size
            images = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device, dtype=torch.float16)
            text_tokens = torch.randint(0, 50000, (batch_size, args.max_text_length), device=device)
            
            # Convert to FP8 for maximum efficiency
            images = images.to(torch.float8_e4m3fn)
            
            # Sample random timesteps
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device)
            
            # Add noise
            noise = torch.randn_like(images)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
                noise_pred = model(noisy_images, text_tokens, timesteps)
                loss = F.mse_loss(noise_pred, noise.to(noise_pred.dtype))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            if rank == 0 and step % args.log_interval == 0:
                print(f"Step {step}/{args.max_steps}, Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.8f}")
            
            if step % args.save_interval == 0 and rank == 0:
                checkpoint = {
                    'model': model.state_dict() if world_size == 1 else model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': step,
                    'args': args
                }
                torch.save(checkpoint, f"checkpoint_step_{step}.pt")
            
            step += 1
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='8b', choices=['8b', '12b', '16b'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--max_text_length', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    
    args = parser.parse_args()
    main(args)
