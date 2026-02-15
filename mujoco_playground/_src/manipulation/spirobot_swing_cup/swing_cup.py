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
"""Spirobot Swing Cup environment - Single Arm Version.

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
  """Default configuration for SpirobotSwingCup."""
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


class SpirobotSwingCup(mjx_env.MjxEnv):
  """Single Spirobot cup task."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides)
    
    # Load original robot XML
    self._xml_path = mjx_env.ROOT_PATH / "manipulation" / "spirobot_swing_cup" / "xmls" / "spirobot" / "ROBOT_ball_joint.xml"
    
    # Load model
    self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
    self._mj_model.opt.timestep = self.sim_dt
    
    # Convert to MJX
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    
    self._post_init()

  def _post_init(self) -> None:
    """Post-initialization to cache body and joint IDs."""
    # Get arm joint IDs (joint_1 to joint_5)
    self._arm_joint_ids = []
    for i in range(self._mj_model.njnt):
      joint_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
      if joint_name and joint_name.startswith("joint_") and joint_name[6:].isdigit():
        joint_num = int(joint_name[6:])
        if 1 <= joint_num <= 5:
          self._arm_joint_ids.append(i)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset environment."""
    rng, reset_rng = jax.random.split(rng)
    
    # Initialize qpos
    qpos = jp.zeros(self._mjx_model.nq)
    # Set ball joint quaternions to identity
    for i in range(self._mjx_model.njnt):
      if self._mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_BALL:
        qposadr = int(self._mj_model.joint(i).qposadr.item())
        qpos = qpos.at[qposadr].set(1.0)
    
    qvel = jp.zeros(self._mjx_model.nv)
    
    # Add small noise to arm joints
    qpos_noise = jax.random.uniform(
        reset_rng, (5,),
        minval=-0.2,
        maxval=0.2
    )
    for i, joint_id in enumerate(self._arm_joint_ids[:5]):
      qposadr = int(self._mj_model.joint(joint_id).qposadr.item())
      qpos = qpos.at[qposadr].set(qpos_noise[i])
    
    # Initial ctrl
    ctrl = jp.zeros(self._mjx_model.nu)
    
    # Create MJX data
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        impl=self._mjx_model.impl.value,
        nconmax=512,
        njmax=1024,
    )
    data = mjx.forward(self._mjx_model, data)
    
    # Initial observation
    obs = self._get_obs(data)
    
    # Metrics
    metrics = {
        "reward/total": jp.zeros(()),
    }
    
    return mjx_env.State(
        data=data,
        obs=obs,
        reward=jp.zeros(()),
        done=jp.zeros(()),
        metrics=metrics,
        info={"rng": rng, "step_count": jp.array(0)}
    )

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Step environment.
    
    Action: 8-dim
    - [0:5]: Arm joint targets
    - [5:8]: Tendon forces
    """
    action = jp.clip(action, -1.0, 1.0)
    
    # Build ctrl array
    ctrl = jp.zeros(self._mjx_model.nu)
    
    # Arm joints: map [-1, 1] to joint limits
    for i, joint_id in enumerate(self._arm_joint_ids[:5]):
      jnt_range = self._mj_model.joint(joint_id).range
      ctrl_val = (action[i] + 1.0) / 2.0 * (jnt_range[1] - jnt_range[0]) + jnt_range[0]
      ctrl = ctrl.at[i].set(ctrl_val)
    
    # Tendon forces: [-1, 1] -> [-100, 100]
    ctrl = ctrl.at[5:8].set(action[5:8] * 100.0)
    
    # Apply action
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    
    # Simple reward
    reward = jp.zeros(())
    
    # Check termination
    done = self._check_done(data)
    
    # Observation
    obs = self._get_obs(data)
    
    # Update metrics
    metrics = {
        "reward/total": reward,
    }
    
    # Update info
    info = state.info
    info["step_count"] = info["step_count"] + 1
    
    return mjx_env.State(
        data=data,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=info
    )

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    """Get observation."""
    # Joint positions
    joint_obs = []
    for joint_id in self._arm_joint_ids[:5]:
      qposadr = int(self._mj_model.joint(joint_id).qposadr.item())
      joint_obs.append(data.qpos[qposadr])
    
    obs = jp.array(joint_obs)
    
    # Pad to fixed size
    target_obs_size = 50
    if obs.shape[0] < target_obs_size:
      obs = jp.pad(obs, (0, target_obs_size - obs.shape[0]))
    else:
      obs = obs[:target_obs_size]
    
    return obs

  def _check_done(self, data: mjx.Data) -> jax.Array:
    """Check if episode is done."""
    nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    return nan_check.astype(float)

  @property
  def xml_path(self) -> str:
    return self._xml_path.as_posix()

  @property
  def action_size(self) -> int:
    return 8  # 5 joints + 3 tendons

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
