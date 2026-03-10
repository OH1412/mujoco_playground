# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Spirobot environment - Single Arm Version.

Simple version that loads the original ROBOT_ball_joint.xml directly.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
  """Default configuration for Spirobot."""
  return config_dict.create(
      ctrl_dt=0.02,  # 50Hz control
      sim_dt=0.002,  # 500Hz simulation
      episode_length=200,
      action_repeat=1,
      vision=False,
      impl="warp",  # Use MuJoCo Warp for spatial tendon support
  )


def domain_randomize(model: mjx.Model, rng: jax.Array) -> mjx.Model:
  """Domain randomization for sim-to-real transfer."""
  return model


# Task implementation removed – this file now only contains helper
# routines related to Spirobot (e.g. visualization or training scripts).
# The actual environment class will be defined elsewhere by the user.

# (If you still need a toy env, recreate it in your own module.)


# previously contained Spirobot class and reset/step logic, removed.
#
# 以下示例片段展示了过去可能存在的环境方法。它们已被注释，
# 若需要可复制到某个具体环境类中：
#
# def _get_obs(self, data: mjx.Data) -> jax.Array:
#     """Get observation."""
#     # Joint positions
#     joint_obs = []
#     for joint_id in self._arm_joint_ids[:5]:
#         qposadr = int(self._mj_model.joint(joint_id).qposadr.item())
#         joint_obs.append(data.qpos[qposadr])
#     obs = jp.array(joint_obs)
#     # Pad to fixed size
#     target_obs_size = 50
#     if obs.shape[0] < target_obs_size:
#         obs = jp.pad(obs, (0, target_obs_size - obs.shape[0]))
#     else:
#         obs = obs[:target_obs_size]
#     return obs
#
# def _check_done(self, data: mjx.Data) -> jax.Array:
#     """Check if episode is done."""
#     nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
#     return nan_check.astype(float)
#
# @property
# def xml_path(self) -> str:
#     return self._xml_path.as_posix()
#
# @property
# def action_size(self) -> int:
#     return 8  # 5 joints + 3 tendons
#
# @property
# def mj_model(self) -> mujoco.MjModel:
#     return self._mj_model
#
# @property
# def mjx_model(self) -> mjx.Model:
#     return self._mjx_model
