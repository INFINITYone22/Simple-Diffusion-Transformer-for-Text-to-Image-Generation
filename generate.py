import torch
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
import transformer_engine.pytorch as te
from model import create_model
from tqdm import tqdm

class DDIMScheduler:
    """DDIM sampling scheduler for fast generation"""
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set inference timesteps"""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, 
             eta: float = 0.0, prev_timestep: int = None) -> torch.Tensor:
        """DDIM sampling step"""
        if prev_timestep is None:
            prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # Compute predicted original sample
        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        
        # Compute variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance.sqrt()
        
        # Compute predicted sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * model_output
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise
        
        return prev_sample

def simple_tokenizer(text: str, vocab_size: int = 50000, max_length: int = 256) -> torch.Tensor:
    """Simple hash-based tokenizer (replace with proper tokenizer)"""
    words = text.lower().split()
    tokens = [hash(word) % vocab_size for word in words]
    
    # Pad or truncate
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    else:
        tokens = tokens[:max_length]
    
    return torch.tensor(tokens, dtype=torch.long)

@torch.no_grad()
def generate_image(model, prompt: str, num_inference_steps: int = 50, 
                  guidance_scale: float = 7.5, image_size: int = 1024,
                  device: torch.device = None) -> Image.Image:
    """Generate image from text prompt"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Tokenize prompt
    text_tokens = simple_tokenizer(prompt).unsqueeze(0).to(device)
    
    # Setup scheduler
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(num_inference_steps, device)
    
    # Initialize random noise
    latents = torch.randn(1, 3, image_size, image_size, device=device, dtype=torch.float8_e4m3fn)
    
    # Enable FP8 inference
    with te.fp8_autocast(enabled=True):
        # Denoising loop
        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Generating")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            timestep_tensor = t.unsqueeze(0).repeat(2)
            
            # Create conditional and unconditional text inputs
            text_input = torch.cat([text_tokens, torch.zeros_like(text_tokens)])
            
            # Predict noise
            with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
                noise_pred = model(latent_model_input, text_input, timestep_tensor)
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            prev_timestep = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else torch.tensor(-1)
            latents = scheduler.step(noise_pred, t, latents, prev_timestep=prev_timestep)
    
    # Convert to PIL Image
    latents = latents.squeeze(0).cpu().float()
    latents = (latents + 1.0) / 2.0  # Denormalize
    latents = torch.clamp(latents, 0.0, 1.0)
    
    # Convert to PIL
    image_array = (latents.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    
    return image

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(config=args.model, image_size=args.image_size)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    model = model.to(device)
    
    # Generate image
    image = generate_image(
        model=model,
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_size=args.image_size,
        device=device
    )
    
    # Save image
    output_path = args.output or f"generated_{hash(args.prompt) % 10000}.png"
    image.save(output_path)
    print(f"Generated image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--model', type=str, default='8b', choices=['8b', '12b', '16b'])
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--image_size', type=int, default=1024, help='Generated image size')
    parser.add_argument('--output', type=str, help='Output image path')
    
    args = parser.parse_args()
    main(args)
