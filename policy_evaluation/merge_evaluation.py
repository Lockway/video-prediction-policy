import json
import os
import shutil
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig
from collections import Counter, defaultdict
import numpy as np

def merge_results_data(all_data):
    """
    Merges multiple 'results.json' style data objects.
    Expects a list of dictionaries where keys are {epoch}_seed{seed}.
    """
    merged = {}
    
    # Group by key (epoch+seed)
    by_key = defaultdict(list)
    for data in all_data:
        for key, val in data.items():
            by_key[key].append(val)
            
    for key, entries in by_key.items():
        if len(entries) == 1:
            merged[key] = entries[0]
            continue
            
        # Combine statistics
        total_eval_sequences = sum(e.get('evaluated_sequences', 1) for e in entries)
        
        # Merge task_info
        merged_task_info = {}
        all_tasks = set()
        for e in entries:
            all_tasks.update(e.get('task_info', {}).keys())
            
        for task in all_tasks:
            success = sum(e.get('task_info', {}).get(task, {}).get('success', 0) for e in entries)
            total = sum(e.get('task_info', {}).get(task, {}).get('total', 0) for e in entries)
            merged_task_info[task] = {'success': success, 'total': total}
            
        # Merge failed_sequences
        merged_failed_sequences = []
        for e in entries:
            merged_failed_sequences.extend(e.get('failed_sequences', []))
        merged_failed_sequences = sorted(list(set(merged_failed_sequences)))
        
        # Recompute avg_seq_len
        # We use evaluated_sequences as weight
        total_seq_len_sum = sum(e.get('avg_seq_len', 0) * e.get('evaluated_sequences', 1) for e in entries)
        avg_seq_len = total_seq_len_sum / total_eval_sequences
        
        # Recompute chain_sr (Success rates for i instructions in a row)
        # chain_sr: { "1": 0.9, "2": 0.8 ... }
        merged_chain_sr = {}
        for i in range(1, 6):
            sum_sr = sum(e.get('chain_sr', {}).get(str(i), 0) * e.get('evaluated_sequences', 1) for e in entries)
            merged_chain_sr[str(i)] = sum_sr / total_eval_sequences
            
        merged[key] = {
            "avg_seq_len": avg_seq_len,
            "chain_sr": merged_chain_sr,
            "task_info": merged_task_info,
            "seed": entries[0].get('seed'),
            "failed_sequences": merged_failed_sequences,
            "evaluated_sequences": total_eval_sequences
        }
        
    return merged

@hydra.main(config_path="../policy_conf", config_name="merge_evaluation")
def main(cfg: DictConfig):
    # Use absolute paths or original CWD to avoid Hydra's directory shifting
    original_cwd = Path(hydra.utils.get_original_cwd())
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = original_cwd / cfg.target_parent_dir / timestamp
    
    print(f"Creating merged directory at: {target_dir}")
    subfolders = ["rollout", "action", "condition", "latent_fake", "latent_real"]
    for folder in subfolders:
        os.makedirs(target_dir / folder, exist_ok=True)
        
    merged_metadata = []
    all_results_data = []
    
    source_dirs = [original_cwd / d for d in cfg.source_dirs]
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
            continue
            
        print(f"Processing {source_dir}...")
        
        # 1. Copy subfolder files
        for folder in subfolders:
            src_folder = source_dir / folder
            if src_folder.exists():
                for item in src_folder.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_dir / folder / item.name)
            else:
                print(f"  Note: Subfolder {folder} not found in {source_dir}")
                
        # 2. Merge metadata.json
        metadata_path = source_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
                # Update paths in metadata
                for entry in data:
                    for key in ["latent_fake_path", "latent_real_path", "condition_path", "action_path"]:
                        if key in entry and entry[key]:
                            old_path = Path(entry[key])
                            # Update to new absolute path
                            entry[key] = str(target_dir / old_path.parent.name / old_path.name)
                    merged_metadata.append(entry)
                    
        # 3. Collect results.json
        results_path = source_dir / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                all_results_data.append(json.load(f))

    # Save merged metadata
    with open(target_dir / "metadata.json", "w") as f:
        json.dump(merged_metadata, f, indent=2)
        
    # Merge and save results
    if all_results_data:
        final_results = merge_results_data(all_results_data)
        with open(target_dir / "results.json", "w") as f:
            json.dump(final_results, f, indent=2)
            
    print(f"Successfully merged {len(source_dirs)} directories into {target_dir}")

if __name__ == "__main__":
    main()
