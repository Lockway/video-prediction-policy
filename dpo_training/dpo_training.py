import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import peft
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import StableVideoDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModelWithProjection
import numpy as np
import math
from einops import rearrange, repeat
from datetime import datetime

logger = get_logger(__name__, log_level="INFO")

class DPODataset(Dataset):
    def __init__(self, metadata_path, depth_consistency_threshold=1.0):
        super().__init__()
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter by success: 0
        failed_entries = [entry for entry in self.metadata if entry.get('success', 1) == 0]
        
        # Sort by depth_consistency descending (highest error first)
        failed_entries.sort(key=lambda x: x.get('depth_consistency', 0), reverse=True)
        
        # Select top percentage
        num_to_keep = max(1, int(len(failed_entries) * depth_consistency_threshold))
        selected_entries = failed_entries[:num_to_keep]
        
        self.pairs = []
        for entry in selected_entries:
            # For failed episodes, we prefer the ground truth (real) over the generated (fake) video
            self.pairs.append({
                'preferred_path': entry['latent_real_path'],
                'rejected_path': entry['latent_fake_path'],
                'task': entry['task'],
                'condition_path': entry['condition_path']
            })
        
        print(f"Total failed entries found: {len(failed_entries)}")
        print(f"Selected DPO pairs (top {depth_consistency_threshold*100}% worst depth consistency): {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load latents
        # Shapes are usually [2, 16, 4, 32, 32] -> [views, frames, channels, h, w]
        p_latents = torch.load(pair['preferred_path'])
        r_latents = torch.load(pair['rejected_path'])
        
        return {
            'preferred_latent': p_latents, # [2, 16, 4, 32, 32]
            'rejected_latent': r_latents,  # [2, 16, 4, 32, 32]
            'text': pair['task'],
            'condition_path': pair['condition_path']
        }

def get_log_probs(unet, noisy_latents, timesteps, encoder_hidden_states, added_time_ids, target_latents, pipeline):
    sigma = timesteps.reshape([-1, 1, 1, 1, 1])
    c_skip = 1 / (sigma**2 + 1)
    c_out =  -sigma / (sigma**2 + 1) ** 0.5
    c_in = 1 / (sigma**2 + 1) ** 0.5
    
    input_latents = torch.cat([c_in * noisy_latents, repeat(target_latents[:, 0], 'b c h w -> b f c h w', f=target_latents.shape[1]) / pipeline.vae.config.scaling_factor], dim=2)
    
    c_noise = (sigma.log() / 4).reshape([-1])
    model_pred = unet(input_latents, c_noise, encoder_hidden_states=encoder_hidden_states, added_time_ids=added_time_ids).sample
    
    target_noise = (noisy_latents - target_latents) / sigma
    loss = F.mse_loss(model_pred, target_noise, reduction='none')
    # Sum over dimensions to get "log prob" per sample
    log_prob = -loss.mean(dim=(1, 2, 3, 4)) 
    return log_prob

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="dpo_training/dpo_training.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    
    # Update output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = os.path.join(cfg.output_dir, timestamp)
    
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="wandb"
    )
    
    set_seed(cfg.seed)
    
    # Load Models
    pipeline = StableVideoDiffusionPipeline.from_pretrained(cfg.pretrained_model_path)
    unet = pipeline.unet
    text_encoder = CLIPTextModelWithProjection.from_pretrained(cfg.text_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_path, use_fast=False)
    
    # Reference model (frozen)
    ref_unet = StableVideoDiffusionPipeline.from_pretrained(cfg.pretrained_model_path).unet
    ref_unet.requires_grad_(False)
    
    # Trainable model with LoRA
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.lora_target_modules),
            lora_dropout=0.05,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
    
    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.learning_rate)
    
    # Dataset
    dataset = DPODataset(cfg.metadata_path, depth_consistency_threshold=cfg.get('depth_consistency_threshold', 1.0))
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    ref_unet = ref_unet.to(accelerator.device)
    text_encoder = text_encoder.to(accelerator.device, dtype=pipeline.vae.dtype)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        accelerator.init_trackers("svd-dpo", config=OmegaConf.to_container(cfg))

    global_step = 0
    for epoch in range(cfg.epochs):
        unet.train()
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Flatten views into batch dimension
                p_latents = rearrange(batch['preferred_latent'], 'b v f c h w -> (b v) f c h w').to(weight_dtype)
                r_latents = rearrange(batch['rejected_latent'], 'b v f c h w -> (b v) f c h w').to(weight_dtype)
                texts = []
                for t in batch['text']:
                    texts.extend([t, t]) # repeat for 2 views
                
                bsz = p_latents.shape[0]
                device = p_latents.device
                
                # Sample timestep and noise (same for p and r)
                timesteps = torch.exp(torch.randn([bsz], device=device) * 1.2 - 1.2) # Sample sigma
                noise = torch.randn_like(p_latents)
                
                p_noisy = p_latents + noise * timesteps.reshape([-1, 1, 1, 1, 1])
                r_noisy = r_latents + noise * timesteps.reshape([-1, 1, 1, 1, 1])
                
                # Encode text
                inputs = tokenizer(texts, padding='max_length', return_tensors="pt", truncation=True, max_length=cfg.max_length).to(device)
                encoder_hidden_states = text_encoder(**inputs).last_hidden_state
                # SVD UNet expects 1024 dimension for cross-attention. 
                # Following step1_train_svd.py logic: concatenate the 512-dim embedding with itself.
                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=-1)
                
                # Added time ids
                added_time_ids = pipeline._get_add_time_ids(cfg.fps, cfg.motion_bucket_id, 0.0, encoder_hidden_states.dtype, bsz, 1, False).to(device)
                
                # Compute log probs
                p_log_probs = get_log_probs(unet, p_noisy, timesteps, encoder_hidden_states, added_time_ids, p_latents, pipeline)
                r_log_probs = get_log_probs(unet, r_noisy, timesteps, encoder_hidden_states, added_time_ids, r_latents, pipeline)
                
                with torch.no_grad():
                    p_ref_log_probs = get_log_probs(ref_unet, p_noisy, timesteps, encoder_hidden_states, added_time_ids, p_latents, pipeline)
                    r_ref_log_probs = get_log_probs(ref_unet, r_noisy, timesteps, encoder_hidden_states, added_time_ids, r_latents, pipeline)
                
                # DPO Loss
                pi_logratios = p_log_probs - r_log_probs
                ref_logratios = p_ref_log_probs - r_ref_log_probs
                
                loss = -F.logsigmoid(cfg.beta * (pi_logratios - ref_logratios)).mean()
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.log_interval == 0:
                    accelerator.log({"loss": loss.item(), "pi_logratio": pi_logratios.mean().item(), "ref_logratio": ref_logratios.mean().item()}, step=global_step)
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")

                if global_step % cfg.save_interval == 0 and accelerator.is_main_process:
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(unet).save_pretrained(save_path)
                    logger.info(f"Saved LoRA adapter to {save_path}")

    # Final save
    if accelerator.is_main_process:
        final_save_path = os.path.join(cfg.output_dir, "final_lora_adapter")
        accelerator.unwrap_model(unet).save_pretrained(final_save_path)
        logger.info(f"Final LoRA adapter saved to {final_save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
