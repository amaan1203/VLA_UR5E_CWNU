from ur5_env import  UR5RobotiqEnv
import gymnasium as gym
from stable_baselines3 import PPO, SAC,A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from dataclasses import dataclass
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch
import cv2
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import queue
import threading
import torch
import os 
import time 
import sys


log_filename = f"test_log.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 

    def flush(self):
        pass


sys.stdout = Logger(log_filename)

print(f"Logging session started. Saving to: {log_filename}")


@dataclass
class VLASample:
    images: dict
    state: np.ndarray
    timestamp: float


def get_front_camera_feed(width=640, height=480):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.5, 0.0, 0.5],
        distance=1.0,
        yaw=90,
        pitch=-30,
        roll=0,
        upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0
    )
    
    (_, _, rgb, _, _) = p.getCameraImage(
        width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    # converting to numpy array
    rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3]
    return rgb_array.astype(np.uint8)

def get_gripper_camera_feed(robot_id, width=320, height=240):
    # index 7 is the 'ee_link' in the ur5e arm
    EE_LINK_INDEX = 7
    
   
    state = p.getLinkState(robot_id, EE_LINK_INDEX, computeForwardKinematics=True)
    pos = state[4]  
    orn = state[5] 
    
    # converting orientation to a rotation matrix to find "Forward" and "Up"
    rot_matrix = p.getMatrixFromQuaternion(orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    

    forward_vec = rot_matrix.dot(np.array([1, 0, 0])) 
    up_vec = rot_matrix.dot(np.array([0, 0, 1]))      
    
    cam_pos = np.array(pos) - (forward_vec * 0.05) + (up_vec * 0.05)
    
    target_pos = np.array(pos) + (forward_vec * 0.2)
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up_vec
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=65, aspect=float(width)/height, nearVal=0.01, farVal=10.0
    )
    
    (_, _, rgb, _, _) = p.getCameraImage(width, height, view_matrix, proj_matrix)
    return np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)

def get_side_camera_feed(width=640, height=480):
    camera_target = [0.5, 0.2, 0.6] 
    
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target,
        distance=1.2,
        yaw=0,            
        pitch=-20,         
        roll=0,
        upAxisIndex=2
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0
    )
    
    (_, _, rgb, _, _) = p.getCameraImage(
        width, height, view_matrix, proj_matrix, 
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3]
    return rgb_array.astype(np.uint8)

def map_vla_to_ur5_absolute(vla_action, current_obs, scale=0.02):
    """
    vla_action: [dx, dy, dz, dr, dp, dy, g] from smolVLA
    current_obs: [ee_x, ee_y, ...] assuming first two are current X, Y
    """
 
    current_x = current_obs[0]
    current_y = current_obs[1]
    
    dx = vla_action[0]
    dy = vla_action[1]
    

    target_x = current_x + (dx * scale)
    target_y = current_y + (dy * scale)
    
    target_x = np.clip(target_x, 0.3, 0.7)
    target_y = np.clip(target_y, -0.3, 0.3)
    
    return np.array([target_x, target_y], dtype=np.float64)



def verify_camera():
    env = UR5RobotiqEnv()
    # print(f"Action Space: {env.action_space}")
    # print(f"High values: {env.action_space.high}")
    # print(f"Low values: {env.action_space.low}")
    env.reset()
    robot_id = env.robot.id 
    
    table_rgb = get_side_camera_feed()
    cv2.imwrite("side_camera.png", cv2.cvtColor(table_rgb, cv2.COLOR_RGB2BGR))
    
    table_rgb = get_front_camera_feed()
    cv2.imwrite("front_camera.png", cv2.cvtColor(table_rgb, cv2.COLOR_RGB2BGR))
    
    table_rgb = get_gripper_camera_feed(robot_id)
    cv2.imwrite("gripper_camera.png", cv2.cvtColor(table_rgb, cv2.COLOR_RGB2BGR))
    env.close()
    
def get_fused_vla_input(robot_id):

    front = get_front_camera_feed(width=320, height=240)
    side = get_side_camera_feed(width=320, height=240)
    gripper = get_gripper_camera_feed(robot_id, width=640, height=240) # Wider to match top row
    
    top_row = np.hstack((front, side))
    
    fused_img = np.vstack((top_row, gripper))
    
    return fused_img


def preprocess_state(env, obs):
    joint_indices = [1, 2, 3, 4, 5, 6] 
    joint_states = p.getJointStates(env.robot.id, joint_indices)
    joint_positions = [state[0] for state in joint_states]
    gripper_state = p.getJointState(env.robot.id, 9)[0]
    
    state_vector = np.concatenate([joint_positions, [gripper_state]]) # length 7
    padded_state = np.zeros(32, dtype=np.float32)
    padded_state[:len(state_vector)] = state_vector
    
    if not hasattr(preprocess_state, "verified"):
        print(f"\n[VERIFICATION] State Projector Initialized")
        print(f" -> Raw State (6 joints + 1 gripper): {state_vector.shape}")
        print(f" -> Padded Vector Shape: {padded_state.shape}")
        preprocess_state.verified = True
        
    return padded_state

   
class VLAPID:
    def __init__(self, model_id, device):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device)
        self.policy.eval()
        instruction = "Pick up the green block, then drop to the target location."
        tokens = self.policy.model.vlm_with_expert.processor.tokenizer(instruction, return_tensors="pt").to(device)
        self.lang_tokens, self.lang_mask = tokens["input_ids"], tokens["attention_mask"].bool()

    @torch.inference_mode()
    def residual(self, fused_image, state):
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        img_t = torch.from_numpy(fused_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        obs_vla = {
            "observation.state": state_t,
            "observation.images.camera1": img_t, 
            "observation.language.tokens": self.lang_tokens,
            "observation.language.attention_mask": self.lang_mask,
        }

        action = self.policy.select_action(obs_vla)
        
        try:
            features = self.policy.model.vlm_with_expert.vision_tower_output 
            latent_norm = torch.norm(features).item()
        except:
            latent_norm = torch.norm(state_t).item()
                
        return action[0, :3].cpu().numpy(), latent_norm


class AsyncVLA:
    def __init__(self, vla):
        self.vla = vla
        self.input_queue = queue.Queue(maxsize=1)
        self.latest_result = np.zeros(3)
        self.latest_norm = 0.0
        self.latest_timestamp = 0.0
        self.new_data_available = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def _inference_loop(self):
        while self.running:
            try:
                images, state, t_captured = self.input_queue.get(timeout=0.1)
                t_inf_start = time.perf_counter()
                result, l_norm = self.vla.residual(images, state)
                inf_time = (time.perf_counter() - t_inf_start) * 1000
                
                with self.lock:
                    self.latest_result, self.latest_norm = result, l_norm
                    self.latest_timestamp, self.new_data_available = t_captured, True
                print(f"  [ASYNC] Inference: {inf_time:.1f}ms")
            except queue.Empty: continue

    def submit(self, images, state, t_captured):
        if self.input_queue.full():
            try: self.input_queue.get_nowait()
            except queue.Empty: pass
        self.input_queue.put_nowait((images, state, t_captured))

    def get_latest(self):
        with self.lock:
            is_new = self.new_data_available
            self.new_data_available = False
            return self.latest_result.copy(), self.latest_norm, self.latest_timestamp, is_new
    
    def stop(self): self.running = False; self.thread.join(); print("AsyncVLA stopped")

def test_algo():
    env = UR5RobotiqEnv()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    vla_model = VLAPID("lerobot/smolvla_base", device)
    async_vla = AsyncVLA(vla_model)
    model = SAC.load(f"ur_robot_sac_7000_steps", env=env)

    obs, info = env.reset()
    robot_id, step_count = env.robot.id, 0
    current_vla_guided_pos = np.array([obs[0], obs[1]], dtype=np.float64)

    try:
        while True:
            step_count += 1
            # sync snapshot every 2 steps
            if step_count % 2 == 0: 
                t_capture = time.perf_counter()
                fused_visual = get_fused_vla_input(robot_id)
                if step_count % 2 == 0:
                      cv2.imwrite("fused_stack.png", cv2.cvtColor(fused_visual, cv2.COLOR_RGB2BGR))
                snapshot = VLASample(
                        images=fused_visual, 
                        state=preprocess_state(env, obs),
                        timestamp=t_capture
                    )
                
                async_vla.submit(snapshot.images, snapshot.state, snapshot.timestamp)

            rl_action, _ = model.predict(obs, deterministic=True)
            vla_raw, l_norm, t_capture, is_new = async_vla.get_latest()
            
            if is_new:
                current_vla_guided_pos = map_vla_to_ur5_absolute(vla_raw, obs, scale=0.01)
                age = (time.perf_counter() - t_capture) * 1000
                print(f"[Step {step_count}] Latent Norm: {l_norm:.4f} | Action Age: {age:.1f}ms")

            hybrid_action = (0.7 * rl_action) + (0.3 * current_vla_guided_pos)
            obs, reward, terminated, truncated, info = env.step(hybrid_action)
            
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        async_vla.stop()
        env.close()

if __name__ == '__main__':
    # verify_camera()
    test_algo()

