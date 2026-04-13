import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# Add Video-Depth-Anything to path
VDA_PATH = Path(__file__).parent.parent / "Video-Depth-Anything"
sys.path.insert(0, str(VDA_PATH))

from video_depth_anything.video_depth import VideoDepthAnything

class DepthConsistencyEvaluator:
    def __init__(self, device="cuda", encoder="vitb", metric=True):
        self.device = device
        self.encoder = encoder
        self.metric = metric
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        
        checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
        checkpoint_path = VDA_PATH / "checkpoints" / f"{checkpoint_name}_{encoder}.pth"
        
        if not checkpoint_path.exists():
            print(f"Warning: Depth model checkpoint not found at {checkpoint_path}")
            self.model = None
            # We don't return here so we can still use calculate_consistency if needed
        else:
            self.model = VideoDepthAnything(**model_configs[encoder], metric=metric)
            self.model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'), strict=True)
            self.model = self.model.to(device).eval()
        
        self.tolerance = 0.05

    @torch.no_grad()
    def estimate_depth(self, frames):
        """
        frames: np.array of shape [F, H, W, 3] (RGB, 0-255)
        returns: np.array of shape [F, H, W]
        """
        if self.model is None:
            # Fallback to dummy depth if model is missing
            return np.ones((frames.shape[0], frames.shape[1], frames.shape[2]))
        
        depths, _ = self.model.infer_video_depth(frames, target_fps=-1, input_size=518, device=self.device)
        return depths

    def calculate_consistency(self, depth0, depth1, K0, K1, R0, R1):
        """
        Calculates reprojection error between two views.
        depth0, depth1: np.array [H, W]
        K0, K1: 3x3 intrinsic matrices
        R0, R1: 4x4 Cam-to-World pose matrices
        """
        H, W = depth0.shape
        
        # World-to-Camera for view 1
        E1 = np.linalg.inv(R1)
        
        # Relative Transformation: Cam0 -> World -> Cam1
        E_rel = E1 @ R0

        # Pixel Coordinates to Camera 0 Space (Back-projection)
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten() + 0.5
        v = v.flatten() + 0.5
        z0 = depth0.flatten()

        valid_mask = z0 > 1e-3
        u, v, z0 = u[valid_mask], v[valid_mask], z0[valid_mask]
        
        if len(z0) == 0:
            return 0.0

        inv_K0 = np.linalg.inv(K0)
        p_homo_img0 = np.stack([u * z0, v * z0, z0], axis=0)
        P_cam0 = inv_K0 @ p_homo_img0
        P_cam0_homo = np.vstack([P_cam0, np.ones((1, P_cam0.shape[1]))])

        # Camera 0 to Camera 1 Space
        P_cam1 = E_rel @ P_cam0_homo
        x1, y1, z1 = P_cam1[:3, :]
        
        valid_z1 = z1 > 1e-3
        x1, y1, z1 = x1[valid_z1], y1[valid_z1], z1[valid_z1]
        
        if len(z1) == 0:
            return 0.0

        # Project to Image 1 Plane
        p_homo_img1 = K1 @ np.stack([x1, y1, z1], axis=0)
        u1 = p_homo_img1[0, :] / z1
        v1 = p_homo_img1[1, :] / z1

        # Rasterization & Comparison
        u1_int = np.round(u1 - 0.5).astype(np.int32)
        v1_int = np.round(v1 - 0.5).astype(np.int32)
        
        valid_bounds = (u1_int >= 0) & (u1_int < W) & (v1_int >= 0) & (v1_int < H)
        u1_int, v1_int, z1 = u1_int[valid_bounds], v1_int[valid_bounds], z1[valid_bounds]

        if len(z1) == 0:
            return 0.0

        # Projective depth map (Z-buffer)
        proj_depth = np.full((H, W), np.inf)
        sort_idx = np.argsort(z1)[::-1]
        proj_depth[v1_int[sort_idx], u1_int[sort_idx]] = z1[sort_idx]

        mask_reprojected = proj_depth < np.inf
        diff = np.abs(proj_depth[mask_reprojected] - depth1[mask_reprojected])
        
        return np.mean(diff)

    def get_intrinsic_matrix(self, fov, width, height):
        foc = height / (2 * np.tan(np.deg2rad(fov) / 2))
        return np.array([
            [foc, 0, width / 2],
            [0, foc, height / 2],
            [0, 0, 1]
        ])

    def euler_to_matrix(self, euler):
        """Euler angles (x, y, z) to rotation matrix"""
        x, y, z = euler
        Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def get_gripper_cam_pose(self, tcp_pos, tcp_orn):
        """
        Calculates gripper camera pose from TCP pose.
        In CALVIN, gripper camera is typically at the same location as TCP but rotated.
        Ref: calvin_env/camera/gripper_camera.py
        """
        # Cam-to-World (Standard CV coords)
        R_tcp = self.euler_to_matrix(tcp_orn)
        
        # OpenGL view matrix is computed with eye=pos, target=pos+rot_y, up=-rot_z
        # This means:
        # standard_z = rot_y (forward)
        # standard_y = rot_z (down)
        # standard_x = rot_x (right)
        
        # We can construct the Cam-to-World matrix directly from the TCP axes
        # rot_x, rot_y, rot_z are the columns of R_tcp
        rot_x = R_tcp[:, 0]
        rot_y = R_tcp[:, 1]
        rot_z = R_tcp[:, 2]
        
        # In CV coords: x=right, y=down, z=forward
        # For CALVIN gripper: x_cv = rot_x, y_cv = rot_z, z_cv = rot_y
        R_cv = np.stack([rot_x, rot_z, rot_y], axis=1)
        
        pose = np.eye(4)
        pose[:3, :3] = R_cv
        pose[:3, 3] = tcp_pos
        return pose

    def get_static_cam_pose(self, look_from, look_at, up_vector):
        """Calculates static camera pose from look_at/from vectors."""
        # This mimics p.computeViewMatrix and then converts to standard CV pose
        forward = np.array(look_at) - np.array(look_from)
        forward /= np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right /= np.linalg.norm(right)
        
        actual_up = np.cross(right, forward)
        actual_up /= np.linalg.norm(actual_up)
        
        # Camera-to-World matrix (Standard CV: x=right, y=down, z=forward)
        # CV y is -actual_up
        R = np.stack([right, -actual_up, forward], axis=1)
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = look_from
        return pose

    def evaluate_chunk(self, frames, actions, initial_robot_obs, static_cfg, gripper_cfg):
        """
        frames: [32, H, W, 3]
        actions: [10, 7] or [16, 7]
        initial_robot_obs: list of 15 values
        static_cfg, gripper_cfg: dicts with fov, width, height, etc.
        """
        num_frames = 16
        static_frames = frames[:num_frames]
        gripper_frames = frames[num_frames:]
        
        all_depths = self.estimate_depth(frames)
        static_depths = all_depths[:num_frames]
        gripper_depths = all_depths[num_frames:]
        
        K_static = self.get_intrinsic_matrix(static_cfg['fov'], static_cfg['width'], static_cfg['height'])
        K_gripper = self.get_intrinsic_matrix(gripper_cfg['fov'], gripper_cfg['width'], gripper_cfg['height'])
        
        R_static = self.get_static_cam_pose(static_cfg['look_from'], static_cfg['look_at'], static_cfg['up_vector'])
        
        # Initial TCP pose
        tcp_pos = np.array(initial_robot_obs[:3])
        tcp_orn = np.array(initial_robot_obs[3:6]) # euler
        
        # CALVIN constants
        max_rel_pos = 0.02
        max_rel_orn = 0.05
        
        errors = []
        for i in range(num_frames):
            # Calculate gripper pose for this frame
            if i > 0 and i <= len(actions):
                # Apply action (relative)
                action = actions[i-1]
                tcp_pos += action[:3] * max_rel_pos
                tcp_orn += action[3:6] * max_rel_orn
            
            R_gripper = self.get_gripper_cam_pose(tcp_pos, tcp_orn)
            
            # Consistency
            err_s2g = self.calculate_consistency(static_depths[i], gripper_depths[i], K_static, K_gripper, R_static, R_gripper)
            err_g2s = self.calculate_consistency(gripper_depths[i], static_depths[i], K_gripper, K_static, R_gripper, R_static)
            errors.append((err_s2g + err_g2s) / 2.0)
            
        return float(np.mean(errors))
