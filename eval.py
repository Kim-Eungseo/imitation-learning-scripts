# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation script for trained policies (Diffusion, ACT, TDMPC) using gymnasium environments."""

from dataclasses import dataclass
from pathlib import Path

import draccus
import gymnasium as gym
import gym_pusht  # noqa: F401 - registers PushT environment
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class


# Mapping from dataset name to gymnasium environment
DATASET_TO_ENV = {
    "lerobot/pusht": "gym_pusht/PushT-v0",
    "lerobot/pusht_image": "gym_pusht/PushT-v0",
    "lerobot/aloha_sim_insertion_human": "gym_aloha/AlohaInsertion-v0",
    "lerobot/aloha_sim_insertion_scripted": "gym_aloha/AlohaInsertion-v0",
    "lerobot/aloha_sim_transfer_cube_human": "gym_aloha/AlohaTransferCube-v0",
    "lerobot/aloha_sim_transfer_cube_scripted": "gym_aloha/AlohaTransferCube-v0",
    "lerobot/xarm_lift_medium": "gym_xarm/XarmLift-v0",
    "lerobot/xarm_lift_medium_replay": "gym_xarm/XarmLift-v0",
    "lerobot/xarm_push_medium": "gym_xarm/XarmPush-v0",
    "lerobot/xarm_push_medium_replay": "gym_xarm/XarmPush-v0",
}


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Path to pretrained model (saved by train.py)
    pretrained_path: str = "outputs/train/lerobot_pusht_diffusion"

    # Environment (auto-detected from model if not specified)
    env_name: str | None = None

    # Evaluation parameters
    n_episodes: int = 10
    max_steps: int = 300
    seed: int = 42

    # Rendering
    render: bool = False
    save_video: bool = False
    video_dir: str = "outputs/videos"

    # Device
    device: str = "cuda"


def load_policy(pretrained_path: str, device: torch.device):
    """Load a pretrained policy from disk."""
    import json

    from lerobot.policies.factory import PolicyProcessorPipeline

    pretrained_path = Path(pretrained_path)

    # Load policy configuration to get the type
    config_path = pretrained_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    policy_type = config.get("type", "diffusion")

    # Get the policy class based on type
    policy_cls = get_policy_class(policy_type)

    # Load the policy
    policy = policy_cls.from_pretrained(pretrained_path)
    policy.eval()
    policy.to(device)

    # Load preprocessor and postprocessor
    preprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, "policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(pretrained_path, "policy_postprocessor.json")

    return policy, preprocessor, postprocessor


def make_env(env_name: str, render: bool = False, seed: int = 42):
    """Create a gymnasium environment."""
    # Use pixels_agent_pos for PushT to get both image and state observations
    if "pusht" in env_name.lower():
        # rgb_array is required for pixel observations, human adds window display
        render_mode = "human" if render else "rgb_array"
        env = gym.make(env_name, render_mode=render_mode, obs_type="pixels_agent_pos")
    else:
        render_mode = "human" if render else None
        env = gym.make(env_name, render_mode=render_mode)
    env.reset(seed=seed)
    return env


# Mapping from gym observation keys to policy observation keys
OBS_KEY_MAPPING = {
    "pixels": "image",
    "agent_pos": "state",
}


def prepare_observation(obs: dict, device: torch.device) -> dict:
    """Convert gymnasium observation to policy input format."""
    batch = {}
    for key, value in obs.items():
        # Map gym observation keys to policy keys
        policy_key = OBS_KEY_MAPPING.get(key, key)

        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value.copy()).float()
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            # Handle image observations (H, W, C) -> (C, H, W)
            if tensor.dim() == 4 and tensor.shape[-1] in [1, 3, 4]:
                tensor = tensor.permute(0, 3, 1, 2)
            batch[f"observation.{policy_key}"] = tensor.to(device)
        else:
            batch[f"observation.{policy_key}"] = torch.tensor([[value]], device=device).float()

    return batch


def rollout_episode(
    env: gym.Env,
    policy: torch.nn.Module,
    preprocessor: torch.nn.Module,
    postprocessor: torch.nn.Module,
    device: torch.device,
    max_steps: int = 300,
) -> dict:
    """Run a single episode and return metrics."""
    obs, info = env.reset()
    policy.reset()

    total_reward = 0.0
    success = False
    step = 0

    for step in range(max_steps):
        # Prepare observation for policy
        batch = prepare_observation(obs, device)
        batch = preprocessor(batch)

        # Get action from policy
        with torch.no_grad():
            action = policy.select_action(batch)

        # Apply postprocessor (unnormalize)
        action_dict = {"action": action}
        action_dict = postprocessor(action_dict)
        action = action_dict["action"]

        # Convert to numpy and remove batch dimension
        action_np = action.squeeze(0).cpu().numpy()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward

        # Check for success
        if "is_success" in info:
            success = info["is_success"]
        elif terminated and reward > 0:
            success = True

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "success": success,
        "length": step + 1,
    }


@draccus.wrap()
def main(cfg: EvalConfig):
    print(f"Evaluating model from: {cfg.pretrained_path}")
    print(f"Number of episodes: {cfg.n_episodes}")
    print(f"Max steps per episode: {cfg.max_steps}")
    print("-" * 50)

    # Set device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy
    print("Loading policy...")
    policy, preprocessor, postprocessor = load_policy(cfg.pretrained_path, device)

    # Determine environment name
    env_name = cfg.env_name
    if env_name is None:
        # Try to infer from pretrained path name
        path_name = cfg.pretrained_path.lower()
        for dataset, env in DATASET_TO_ENV.items():
            if dataset.replace("/", "_").replace("lerobot_", "") in path_name:
                env_name = env
                break
        if env_name is None:
            # Default to pusht
            env_name = "gym_pusht/PushT-v0"
    print(f"Using environment: {env_name}")

    # Create environment
    env = make_env(env_name, render=cfg.render, seed=cfg.seed)

    # Set up video recording
    if cfg.save_video:
        video_dir = Path(cfg.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda x: True,
            name_prefix="eval",
        )

    # Run evaluation episodes
    print("\nRunning evaluation...")
    results = []

    for episode in range(cfg.n_episodes):
        result = rollout_episode(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            max_steps=cfg.max_steps,
        )
        results.append(result)
        print(
            f"Episode {episode + 1}/{cfg.n_episodes}: "
            f"reward={result['reward']:.2f}, "
            f"success={result['success']}, "
            f"length={result['length']}"
        )

    # Compute statistics
    rewards = [r["reward"] for r in results]
    successes = [r["success"] for r in results]
    lengths = [r["length"] for r in results]

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate: {np.mean(successes) * 100:.1f}%")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print("=" * 50)

    env.close()

    # Return results for programmatic use
    return {
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "episodes": results,
    }


if __name__ == "__main__":
    main()
