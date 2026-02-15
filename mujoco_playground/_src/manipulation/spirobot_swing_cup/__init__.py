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
"""Spirobot Swing Cup task.

 migrated from PKU Bi-DexHands project.
"""

from mujoco_playground._src.manipulation.spirobot_swing_cup import swing_cup


def default_config():
  return swing_cup.default_config()


def domain_randomize(model, rng):
  return swing_cup.domain_randomize(model, rng)
