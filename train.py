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

"""Training script for multiple policies (Diffusion, ACT, TDMPC) using draccus."""

from dataclasses import dataclass
from pathlib import Path

import draccus  # noqa: F401
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors


SUPPORTED_POLICIES = ["diffusion", "act", "tdmpc"]


@dataclass
class TrainConfig:
    """Training configuration."""

    # Policy selection: diffusion, act, tdmpc
    policy: str = "diffusion"

    # Dataset
    dataset_name: str = "lerobot/pusht"

    # Training parameters
    training_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_workers: int = 4
    log_freq: int = 1

    # Output
    output_dir: str = "outputs/train"

    # Device
    device: str = "cuda"


def get_policy_and_config(
    policy_type: str,
    input_features: dict,
    output_features: dict,
):
    """Create policy configuration and model based on policy type."""
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
        policy = DiffusionPolicy(cfg)

    elif policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = ACTConfig(input_features=input_features, output_features=output_features)
        policy = ACTPolicy(cfg)

    elif policy_type == "tdmpc":
        from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        cfg = TDMPCConfig(input_features=input_features, output_features=output_features)
        policy = TDMPCPolicy(cfg)

    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {SUPPORTED_POLICIES}")

    return cfg, policy


def get_delta_timestamps(cfg, dataset_metadata):
    """Get delta timestamps based on policy configuration.

    This follows lerobot's resolve_delta_timestamps logic:
    - Only set delta_timestamps for features that have corresponding delta_indices
    - If delta_indices is None, don't include that feature (no temporal dimension)
    """
    delta_timestamps = {}

    # Observation timestamps - only set if observation_delta_indices is not None
    if cfg.observation_delta_indices is not None:
        obs_timestamps = [i / dataset_metadata.fps for i in cfg.observation_delta_indices]
        # Set for all observation features in the dataset
        for key in dataset_metadata.features:
            if key.startswith("observation."):
                delta_timestamps[key] = obs_timestamps

    # Action timestamps
    if cfg.action_delta_indices is not None:
        delta_timestamps["action"] = [i / dataset_metadata.fps for i in cfg.action_delta_indices]

    # Reward timestamps (for TDMPC)
    if hasattr(cfg, "reward_delta_indices") and cfg.reward_delta_indices is not None:
        delta_timestamps["next.reward"] = [i / dataset_metadata.fps for i in cfg.reward_delta_indices]

    # Return None if empty (lerobot convention)
    return delta_timestamps if delta_timestamps else None


@draccus.wrap()
def main(cfg: TrainConfig):
    print(f"Training with policy: {cfg.policy}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Training steps: {cfg.training_steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("-" * 50)

    # Create output directory
    output_directory = Path(cfg.output_dir) / f"{cfg.dataset_name.replace('/', '_')}_{cfg.policy}"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset metadata for policy configuration
    dataset_metadata = LeRobotDatasetMetadata(cfg.dataset_name)
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Create policy and configuration
    policy_cfg, policy = get_policy_and_config(
        cfg.policy,
        input_features=input_features,
        output_features=output_features,
    )
    policy.train()
    policy.to(device)

    # Create preprocessor and postprocessor
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=dataset_metadata.stats)

    # Get delta timestamps for the dataset
    delta_timestamps = get_delta_timestamps(policy_cfg, dataset_metadata)
    print(f"Delta timestamps: {delta_timestamps}")

    # Create dataset
    dataset = LeRobotDataset(cfg.dataset_name, delta_timestamps=delta_timestamps)

    # Create optimizer and dataloader
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg.log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= cfg.training_steps:
                done = True
                break

    # Save checkpoint
    print(f"Saving model to {output_directory}")
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    print("Training completed!")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
