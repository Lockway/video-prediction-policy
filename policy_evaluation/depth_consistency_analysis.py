import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import cv2

# Add current directory to path
sys.path.append(os.getcwd())

from policy_evaluation.depth_consistency_utils import DepthConsistencyEvaluator

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Depth Consistency Analysis')
    parser.add_argument('--log_dir', type=str, help='Path to the log directory (e.g., checkpoint/dp-calvin/logs/2026-04-13_17-17-05/)')
    parser.add_argument('--config', type=str, default='policy_conf/depth_analysis_conf.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    
    log_dir = Path(args.log_dir if args.log_dir else config.get('log_dir', ''))
    if not log_dir.exists():
        print(f"Error: Log directory not found at {log_dir}")
        return

    # In your setup, metadata.json is in the log_dir
    metadata_path = log_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {log_dir}")
        return

    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    device = config.get('device', 'cuda')
    encoder = config.get('encoder', 'vitb')
    metric = config.get('metric', True)
    
    print(f"Initializing DepthConsistencyEvaluator on {device}...")
    evaluator = DepthConsistencyEvaluator(device=device, encoder=encoder, metric=metric)
    
    static_cfg = config['static_camera']
    gripper_cfg = config['gripper_camera']

    updated = False
    for entry in tqdm(metadata_list, desc="Analyzing depth consistency"):
        if "depth_consistency" in entry and not config.get('overwrite', False):
            continue
            
        # Paths in metadata are usually relative or need fixing
        video_name = Path(entry['video_path']).name
        action_name = Path(entry['action_path']).name
        
        video_path = log_dir / "videos" / video_name
        action_path = log_dir / "action" / action_name
        
        if not video_path.exists():
            print(f"Warning: Video not found at {video_path}")
            continue
            
        if not action_path.exists():
            print(f"Warning: Action file not found at {action_path}")
            continue

        if 'robot_obs' not in entry:
            print(f"Warning: robot_obs missing in metadata for {video_path}")
            continue

        # Load video frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        
        if len(frames) != 32:
            print(f"Warning: Expected 32 frames, got {len(frames)} for {video_path}")
            continue
            
        frames = np.stack(frames) 
        
        # Load predicted actions
        actions = torch.load(action_path)
        if isinstance(actions, torch.Tensor):
            actions = actions.squeeze().cpu().numpy()
        
        # Calculate consistency
        score = evaluator.evaluate_chunk(
            frames, 
            actions, 
            entry['robot_obs'], 
            static_cfg, 
            gripper_cfg
        )
        
        entry['depth_consistency'] = float(score)
        updated = True

    if updated:
        # Save back to metadata.json
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        print(f"Finished analysis. Results updated in {metadata_path}")
    else:
        print("No updates made.")

if __name__ == "__main__":
    main()
