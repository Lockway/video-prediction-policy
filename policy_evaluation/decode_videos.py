import argparse
import json
import os
from pathlib import Path
import sys
import torch
import hydra
from moviepy.editor import ImageSequenceClip
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import time

# Add parent directory to sys.path to allow imports from policy_models
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from policy_models.utils.utils import get_last_checkpoint

def decode_and_save_video(model, latent_path, output_path):
    device = model.device
    vae = model.TVP_encoder.pipeline.vae
    scaling_factor = vae.config.scaling_factor
    
    # Load latents
    latents = torch.load(latent_path, map_location=device)
    # Expected shape: [2, 16, 4, H/8, W/8] (static and gripper)
    
    latents_to_decode = latents / scaling_factor
    
    with torch.no_grad():
        b, f, c, h, w = latents_to_decode.shape
        # Flatten batch and frames for VAE decoder: (B*F, C, H, W)
        latents_flat = rearrange(latents_to_decode, "b f c h w -> (b f) c h w")
        
        # Decode in chunks to avoid OOM
        decoded_list = []
        chunk_size = 8
        for i in range(0, latents_flat.shape[0], chunk_size):
            chunk = latents_flat[i : i + chunk_size]
            # Use the pipeline's VAE decoder
            decoded = vae.decode(chunk, num_frames=f).sample
            decoded_list.append(decoded)
        
        frames = torch.cat(decoded_list, dim=0) # [B*F, 3, H, W]
        # Post-process from [-1, 1] to [0, 1]
        frames = (frames / 2 + 0.5).clamp(0, 1)
        
        # Reshape to [B, F, H, W, C] for video generation
        frames = rearrange(frames, "(b f) c h w -> b f h w c", b=b, f=f)
        frames = (frames.cpu().numpy() * 255).astype(np.uint8)
        
    # frames[0] is static, frames[1] is gripper
    static_frames = frames[0]
    gripper_frames = frames[1]
    
    # Concatenate side-by-side (1x2 grid)
    combined_frames = np.concatenate([static_frames, gripper_frames], axis=2) # [F, H, W*2, 3]
    
    # Create and write video
    clip = ImageSequenceClip(list(combined_frames), fps=4)
    clip.write_videofile(str(output_path), codec="libx264", logger=None)

def main():
    parser = argparse.ArgumentParser(description="Decode AI-generated latents from failed evaluation rollouts.")
    parser.add_argument("--datetime", type=str, required=True, help="The datetime string of the log folder (e.g., 2026-04-20_18-58-32)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--action_model_folder", type=str, default="checkpoint/dp-calvin", help="Path to the action model folder")
    
    args = parser.parse_args()

    log_dir = Path(args.action_model_folder) / "logs" / args.datetime
    metadata_path = log_dir / "metadata.json"
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return

    video_out_dir = log_dir / "videos"
    video_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Filter for entries where success: 0
    failed_entries = [e for e in metadata if e.get("success") == 0]
    
    if not failed_entries:
        print(f"No failed tasks found in {metadata_path}")
        return

    print(f"Found {len(failed_entries)} failed tasks. Initializing model for decoding...")

    # Use Hydra to load the configuration
    from hydra import compose, initialize
    # Ensure config_path is relative to this file
    config_rel_path = "../policy_conf"
    
    # Initialize Hydra once
    try:
        initialize(config_path=config_rel_path, version_base=None)
    except Exception as e:
        # Hydra might already be initialized in some environments
        pass
        
    cfg = compose(config_name="calvin_evaluate_all.yaml")
    
    # Update config with correct folder if different
    cfg.train_folder = args.action_model_folder
    
    # Find the latest checkpoint
    checkpoint = get_last_checkpoint(Path(args.action_model_folder))
    if checkpoint is None:
        print(f"Error: No checkpoint found in {args.action_model_folder}")
        return
        
    print(f"Loading checkpoint: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location='cpu')
    
    # Instantiate model
    device = torch.device(f"cuda:{args.device}")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(state_dict['model'], strict=False)
    model.freeze()
    model = model.to(device)
    model.process_device()
    model.eval()

    print("Starting decoding...")
    for entry in tqdm(failed_entries):
        latent_rel_path = entry.get("latent_fake_path")
        if not latent_rel_path:
            continue
            
        latent_path = Path(latent_rel_path)
        # If the path in metadata is relative or needs adjustment, we handle it
        if not latent_path.exists():
            # Try finding it relative to the log_dir
            latent_path = log_dir / "latent_fake" / latent_path.name
            
        if not latent_path.exists():
            print(f"Warning: Could not find latent file {latent_path.name}")
            continue
            
        # Filename format: {tag}_{i}_{subtask}_seed{seed}_step{step}_video.mp4
        base_name = latent_path.stem.replace("_latents_fake", "")
        video_filename = f"{base_name}_video.mp4"
        output_path = video_out_dir / video_filename
        
        if output_path.exists():
            continue
            
        try:
            decode_and_save_video(model, latent_path, output_path)
        except Exception as e:
            print(f"Error decoding {latent_path.name}: {e}")

    print(f"Done! Decoded videos are in: {video_out_dir}")

if __name__ == "__main__":
    main()
