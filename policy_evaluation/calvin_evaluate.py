from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())
import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist
from moviepy.editor import ImageSequenceClip

from omegaconf import ListConfig
from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint
from policy_models.rollout.rollout_video import RolloutVideo

logger = logging.getLogger(__name__)


VIDEO_TAG = "sequence"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.train_folder)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        if "=" in checkpoint.stem:
            epoch = checkpoint.stem.split("=")[1]
        else:
            epoch = checkpoint.stem
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info, "seed": cfg.seed}
        # wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
        # Error: You must call wandb.init() before wandb.log()
        current_data[f"{epoch}_seed{cfg.seed}"] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations
    
    if save_dir is not None:
        for folder in ["rollout", "videos", "action", "condition", "latent"]:
            os.makedirs(save_dir / folder, exist_ok=True)

    # video stuff
    if num_videos > 0:
        model.record_video = True
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir / "rollout" if save_dir else None,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(cfg.num_sequences)

    results = []
    plans = defaultdict(list)
    model.metadata_list = []

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = i < num_videos
        result = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, save_dir=save_dir
        )
        results.append(result)
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        if result < cfg.record_ths and record:
            rollout_video._log_currentvideos_to_file(f"{i}_seed{cfg.seed}", save_as_video=True)

    #if num_videos > 0:
    #    print('save_video_2:',rollout_video.save_dir)
    #    # log rollout videos
    #    rollout_video._log_videos_to_file(0, save_as_video=True)
    
    if save_dir is not None:
        metadata_path = save_dir / "metadata.json"
        existing_metadata = []
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    existing_metadata = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
        
        # Avoid duplicate entries if running the same seed twice
        # We can check by video_path or some unique ID
        seen_videos = {entry.get("video_path") for entry in existing_metadata}
        new_entries = [entry for entry in model.metadata_list if entry.get("video_path") not in seen_videos]
        
        combined_metadata = existing_metadata + new_entries
        with open(metadata_path, "w") as f:
            json.dump(combined_metadata, f, indent=2)
            
    return results, plans


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i, save_dir=None
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence) if cfg.use_caption else ""
        rollout_video.new_video(tag=VIDEO_TAG, caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask in eval_sequence:
        if record:
            rollout_video.new_subtask()
        success = rollout(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video, i, save_dir=save_dir)
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


import threading
import queue

class BackgroundSaver:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            func, args, kwargs = item
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"Error in background saver: {e}")
            self.queue.task_done()

    def save(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))

    def wait(self):
        self.queue.join()

bg_saver = BackgroundSaver()

def save_video_and_data(base_name, ai_frames, data, save_dir, cfg_seed, subtask, tag_clean, i, timestamp):
    # [2] videos: ai-generated videos with different views, temporally concatenated
    concat_frames = torch.cat([ai_frames[0], ai_frames[1]], dim=0) # [32, 3, H, W]
    
    video_filename = f"{base_name}.mp4"
    video_path = save_dir / "videos" / video_filename
    
    video_np = (concat_frames.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    clip = ImageSequenceClip(list(video_np), fps=30)
    clip.write_videofile(str(video_path), codec='libx264', logger=None)
    
    # [3] actions, latents, condition: .pt files
    action_filename = f"{base_name}_actions.pt"
    latent_filename = f"{base_name}_latents.pt"
    condition_filename = f"{base_name}_condition.pt"
    
    torch.save(data['actions'].cpu(), save_dir / "action" / action_filename)
    torch.save(data['latents'].cpu(), save_dir / "latent" / latent_filename)
    torch.save(data['condition'].cpu(), save_dir / "condition" / condition_filename)

def rollout(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None, i=0, save_dir=None):
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    current_sequence_metadata = []
    cached_bottom_row = None

    for step in range(cfg.ep_len):
        action = model.step(obs, goal)
        
        # Check for new chunk
        if record and save_dir is not None and model.last_chunk_data is not None and model.rollout_step_counter == 1:
            chunk_start_step = step
            data = model.last_chunk_data
            
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            tag_clean = VIDEO_TAG.replace("/", "_")
            base_name = f"{tag_clean}_{i}_{subtask}_seed{cfg.seed}_step{chunk_start_step}"
            
            # Use background saver for video and data
            bg_saver.save(save_video_and_data, base_name, data['ai_video_frames'], data, save_dir, cfg.seed, subtask, tag_clean, i, timestamp)
            
            video_filename = f"{base_name}.mp4"
            video_path = save_dir / "videos" / video_filename
            
            action_filename = f"{base_name}_actions.pt"
            latent_filename = f"{base_name}_latents.pt"
            condition_filename = f"{base_name}_condition.pt"
            
            metadata_entry = {
                "original_video_path": str(Path("rollout") / f"{tag_clean}_{i}_seed{cfg.seed}.mp4"),
                "dataset_source": "CALVIN",
                "task": subtask,
                "sequence": i,
                "step": chunk_start_step,
                "video_path": str(video_path),
                "latent_path": str(save_dir / "latent" / latent_filename),
                "condition_path": str(save_dir / "condition" / condition_filename),
                "action_path": str(save_dir / "action" / action_filename),
                "seed": cfg.seed,
                "num_frames": 32, # 16 * 2 views
                "time": timestamp,
                "success": 0, # Default to 0, will update if task succeeds in this chunk
                "robot_obs": obs["robot_obs_raw"].cpu().numpy().tolist(), # Initial robot state for this chunk
            }
            current_sequence_metadata.append(metadata_entry)
            model.metadata_list.append(metadata_entry)
            cached_bottom_row = None # Reset cache when new chunk starts

        obs, _, _, current_info = env.step(action)
        
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if record and save_dir is not None:
                # Update all chunks in this subtask as successful if the subtask is completed
                for entry in current_sequence_metadata:
                    entry["success"] = 1
            
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record and cfg.use_caption:
                rollout_video.add_language_instruction(lang_annotation)
            return True
        
        if record:
            # update video
            static_rgb = obs["rgb_obs"]["rgb_static"]
            gripper_rgb = obs["rgb_obs"]["rgb_gripper"]
            
            if model.last_ai_frame is not None:
                static_ai = model.last_ai_frame['static']
                gripper_ai = model.last_ai_frame['gripper']

                # Convert from [0, 1] to [-1, 1] to match real frame normalization expected by RolloutVideo
                static_ai = static_ai * 2 - 1
                gripper_ai = gripper_ai * 2 - 1

                if static_rgb.shape[-2:] != gripper_rgb.shape[-2:]:
                    gripper_rgb = F.interpolate(gripper_rgb, size=static_rgb.shape[-2:])
                
                if static_ai.shape[-2:] != static_rgb.shape[-2:]:
                    static_ai = F.interpolate(static_ai.unsqueeze(0), size=static_rgb.shape[-2:]).squeeze(0)
                if gripper_ai.shape[-2:] != static_rgb.shape[-2:]:
                    gripper_ai = F.interpolate(gripper_ai.unsqueeze(0), size=static_rgb.shape[-2:]).squeeze(0)
                
                while static_ai.ndim < static_rgb.ndim:
                    static_ai = static_ai.unsqueeze(0)
                while gripper_ai.ndim < static_rgb.ndim:
                    gripper_ai = gripper_ai.unsqueeze(0)
                
                cached_bottom_row = torch.cat([static_ai, gripper_ai], dim=-1)

                top_row = torch.cat([static_rgb, gripper_rgb], dim=-1)
                combined_rgb = torch.cat([top_row, cached_bottom_row], dim=-2)
                rollout_video.update(combined_rgb)
            else:
                top_row = torch.cat([static_rgb, gripper_rgb], dim=-1)
                rollout_video.update(torch.cat([top_row, torch.zeros_like(top_row)], dim=-2))

    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record and cfg.use_caption:
        rollout_video.add_language_instruction(lang_annotation)
    return False
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record and cfg.use_caption:
        rollout_video.add_language_instruction(lang_annotation)
    return False


#@hydra.main(config_path="../policy_conf", config_name="calvin_evaluate_all")
def main(cfg):
    log_wandb = cfg.log_wandb
    torch.cuda.set_device(cfg.device)
    seed_everything(0, workers=True)  # type:ignore
    # evaluate a custom model
    checkpoints = [get_last_checkpoint(Path(cfg.train_folder))]
    lang_embeddings = None
    env = None
    results = {}
    plans = {}

    print('train_folder',cfg.train_folder)

    for checkpoint in checkpoints:
        if checkpoint is None:
            print(f"No checkpoint found in {cfg.train_folder}")
            continue
        print(cfg.device)
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=cfg.device,
            cfg=cfg,
        )
        
        ckpt = checkpoint
        print(f"Loading model from {ckpt}")
        state_dict = torch.load(ckpt, map_location='cpu')
        #print('state_dict_key:', state_dict['model'].keys())
        device = torch.device(f"cuda:{cfg.device}")
        #c = []
        #hydra.initialize(config_path="../../conf")
        #hydra.main(config_name="config_abc.yaml")(lambda x: c.append(x))()
        model = hydra.utils.instantiate(cfg.model)
        #model_state_dict = model.state_dict()
        #model_state_dict.update(state_dict['model'])
        #model.load_state_dict(model_state_dict)
        model.load_state_dict(state_dict['model'],strict = False)
        model.freeze()
        model = model.cuda(device)
        #model = load_model()
        print(cfg.num_sampling_steps, cfg.sampler_type, cfg.multistep, cfg.sigma_min, cfg.sigma_max, cfg.noise_scheduler)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep
        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model.process_device()
        model.eval()
        if log_wandb:
            log_dir = get_log_dir(cfg.train_folder)
            os.makedirs(log_dir / "wandb", exist_ok=True)
            seeds = list(cfg.seed) if isinstance(cfg.seed, (list, tuple, ListConfig)) else [cfg.seed]
            for seed in seeds:
                print(f"Evaluating seed: {seed}")
                cfg.seed = seed
                model.seed = seed

                seed_results = {}
                seed_plans = {}
                seed_results[checkpoint], seed_plans[checkpoint] = evaluate_policy(model, env, lang_embeddings, cfg, num_videos=cfg.num_videos, save_dir=Path(log_dir))
                print_and_save(seed_results, seed_plans, cfg, log_dir=log_dir)
            #run.finish()


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_model_path", type=str, default="")
    parser.add_argument("--action_model_folder", type=str, default="")
    parser.add_argument("--clip_model_path", type=str, default="")
    parser.add_argument("--calvin_abc_dir", type=str, default="")
    
    args = parser.parse_args()
    
    with initialize(config_path="../policy_conf", job_name="calvin_evaluate_all.yaml"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")
    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.root_data_dir = args.calvin_abc_dir
    main(cfg)

    # python policy_evaluation/calvin_evaluate.py --video_model_path /home/disk2/gyj/hyc_ckpt/svd_2camera/checkpoint-100000 --action_model_folder /home/disk2/gyj/hyccode/Video-Prediction-Policy/checkpoint/alllayer1 --clip_model_path /home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32 --calvin_abc_dir /home/disk2/gyj/task_ABC_D
