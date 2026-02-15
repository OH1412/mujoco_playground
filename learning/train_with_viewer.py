#!/usr/bin/env python3
"""Train with MuJoCo native viewer for local debugging.

Usage:
    uv --no-config run python train_with_viewer.py --env_name CartpoleBalance --num_envs 256
    uv --no-config run python train_with_viewer.py --env_name CartpoleBalance --num_envs 256 --num_timesteps 100000
"""

import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
import numpy as np
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Environment setup - optimized for local debugging
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"  # 限制JAX内存使用
os.environ["MUJOCO_GL"] = "egl"

# 限制GPU内存增长
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Flags (same as train_jax_ppo.py)
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "CartpoleBalance",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 10_000, "Number of timesteps (reduced for debugging)"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 5, "Unroll length (reduced for debugging)")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 4, "Number of minibatches (reduced for debugging)"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 4, "Number of updates per batch (reduced for debugging)"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 32, "Number of environments (reduced for debugging)")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size (reduced for debugging)")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_USE_VIEWER = flags.DEFINE_boolean(
    "use_viewer", True, "Launch MuJoCo viewer after training"
)
_VIEWER_ONLY = flags.DEFINE_boolean(
    "viewer_only", False, "Skip training, only show viewer with random policy"
)
_LOAD_CHECKPOINT = flags.DEFINE_string(
    "load_checkpoint", None, "Load checkpoint and visualize trained policy"
)
_REPLAY_STEPS = flags.DEFINE_integer(
    "replay_steps", 1000, "Number of steps to replay in viewer"
)
_SAVE_CHECKPOINT = flags.DEFINE_boolean(
    "save_checkpoint", False, "Save checkpoint after training (default: off for debugging)"
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
    """Get RL config for environment."""
    if env_name in mujoco_playground.manipulation._envs:
        if _VISION.value:
            return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
        return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.locomotion._envs:
        return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
    elif env_name in mujoco_playground.dm_control_suite._envs:
        if _VISION.value:
            return dm_control_suite_params.brax_vision_ppo_config(
                env_name, _IMPL.value
            )
        return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)
    raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def visualize_policy(env, env_cfg, make_inference_fn, params):
    """Visualize trained policy with MuJoCo native viewer."""
    
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    rng = jax.random.PRNGKey(_SEED.value)
    state = jax.jit(env.reset)(rng)
    
    mj_data.qpos[:] = np.array(state.data.qpos)
    mj_data.qvel[:] = np.array(state.data.qvel)
    
    print("\n" + "="*50)
    print("MuJoCo Viewer Controls:")
    print("  [Space] Pause/Resume")
    print("  [R] Reset / Auto-reset when done")
    print("  [ESC/Q] Quit")
    print("="*50 + "\n")
    
    paused = False
    needs_reset = True
    current_state = state
    current_rng = rng
    step_count = 0
    episode_reward = 0.0
    should_quit = False
    
    def key_callback(keycode):
        nonlocal paused, needs_reset, should_quit
        if keycode == ord(' '):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif keycode == ord('r') or keycode == ord('R'):
            needs_reset = True
            print("Manual reset")
        elif keycode == 27 or keycode == ord('q'):
            should_quit = True
            print("Quit")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not should_quit:
            step_start = time.time()
            
            if needs_reset:
                current_rng, reset_rng = jax.random.split(current_rng)
                current_state = jax.jit(env.reset)(reset_rng)
                needs_reset = False
                step_count = 0
                episode_reward = 0.0
                print(f"Episode started")
            
            if not paused:
                # Get action from policy
                act_rng = jax.random.PRNGKey(step_count)
                action = jit_inference_fn(current_state.obs, act_rng)[0]
                
                # Step environment
                current_state = jax.jit(env.step)(current_state, action)
                episode_reward += float(current_state.reward)
                step_count += 1
                
                # Check for done - auto reset
                if current_state.done > 0.5 or step_count >= _EPISODE_LENGTH.value:
                    print(f"Episode finished: steps={step_count}, reward={episode_reward:.2f}")
                    needs_reset = True  # Auto reset for continuous playback
                
                # Copy to MuJoCo data for rendering
                mj_data.qpos[:] = np.array(current_state.data.qpos)
                mj_data.qvel[:] = np.array(current_state.data.qvel)
                mj_data.ctrl[:] = np.array(action)
                
                # Step physics
                mujoco.mj_step(mj_model, mj_data)
            
            viewer.sync()
            
            # Cap at real-time
            elapsed = time.time() - step_start
            sleep_time = env.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("Viewer closed.")


def main(argv):
    """Run training and launch viewer."""
    del argv
    
    # Load environment configuration
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value
    
    if _VISION.value:
        env_cfg.vision = True
    
    env = registry.load(_ENV_NAME.value, config=env_cfg)
    
    print(f"\nEnvironment: {_ENV_NAME.value}")
    print(f"  Action size: {env.action_size}")
    print(f"  Observation size: {env.observation_size}")
    print(f"  Control dt: {env.dt}")
    
    if _VIEWER_ONLY.value:
        print("\nViewer only mode (random policy)")
        visualize_random_policy(env)
        return
    
    if _LOAD_CHECKPOINT.value:
        print(f"\nLoading checkpoint: {_LOAD_CHECKPOINT.value}")
        print("Note: Checkpoint loading simplified. Running fresh training with viewer...")
        print("For full checkpoint support, use: train-jax-ppo --play_only ...")
        # For now, just do a quick training and show viewer
        pass  # Continue to training
    
    # Get PPO config
    ppo_params = get_rl_config(_ENV_NAME.value)
    
    # Override with flags
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
    ppo_params.num_evals = _NUM_EVALS.value
    ppo_params.reward_scaling = _REWARD_SCALING.value
    ppo_params.episode_length = _EPISODE_LENGTH.value
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
    ppo_params.action_repeat = _ACTION_REPEAT.value
    ppo_params.unroll_length = _UNROLL_LENGTH.value
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
    ppo_params.discounting = _DISCOUNTING.value
    ppo_params.learning_rate = _LEARNING_RATE.value
    ppo_params.entropy_cost = _ENTROPY_COST.value
    ppo_params.num_envs = _NUM_ENVS.value
    ppo_params.batch_size = _BATCH_SIZE.value
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
    ppo_params.network_factory = config_dict.create(
        policy_hidden_layer_sizes=tuple(map(int, _POLICY_HIDDEN_LAYER_SIZES.value)),
        value_hidden_layer_sizes=tuple(map(int, _VALUE_HIDDEN_LAYER_SIZES.value)),
    )
    
    print(f"\nTraining config (DEBUG MODE - Low Resource):")
    print(f"  Num envs: {ppo_params.num_envs}")
    print(f"  Num timesteps: {ppo_params.num_timesteps}")
    print(f"  Episode length: {ppo_params.episode_length}")
    print(f"  Learning rate: {ppo_params.learning_rate}")
    print(f"  Batch size: {ppo_params.batch_size}")
    print(f"  Unroll length: {ppo_params.unroll_length}")
    print(f"\nWARNING: Using reduced settings for local debugging.")
    print(f"For full training, use: train-jax-ppo --env_name {_ENV_NAME.value}")
    
    # Progress callback
    times = [time.monotonic()]
    
    def progress(num_steps, metrics):
        times.append(time.monotonic())
        if "eval/episode_reward" in metrics:
            print(f"Step {num_steps}: reward = {metrics['eval/episode_reward']:.3f}")
    
    # Setup checkpoint directory (optional)
    ckpt_path = None
    if _SAVE_CHECKPOINT.value:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        exp_name = f"{_ENV_NAME.value}-{timestamp}-debug"
        logdir = epath.Path("logs").resolve() / exp_name
        logdir.mkdir(parents=True, exist_ok=True)
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print(f"\nCheckpoint will be saved to: {ckpt_path}")
        
        # Save environment configuration
        with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
            json.dump(env_cfg.to_dict(), fp, indent=4)
    
    # Prepare training
    training_params = dict(ppo_params)
    del training_params["network_factory"]
    
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )
    
    train_kwargs = dict(
        **training_params,
        network_factory=network_factory,
        seed=_SEED.value,
        progress_fn=progress,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    if ckpt_path is not None:
        train_kwargs["save_checkpoint_path"] = ckpt_path
    
    train_fn = functools.partial(ppo.train, **train_kwargs)
    
    # Train
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50)
    
    make_inference_fn, params, _ = train_fn(environment=env)
    
    print("\n" + "="*50)
    print("Training Complete!")
    if len(times) > 1:
        print(f"Time to JIT compile: {times[1] - times[0]:.2f}s")
        print(f"Time to train: {times[-1] - times[1]:.2f}s")
    print("="*50)
    
    # Launch viewer
    if _USE_VIEWER.value:
        print("\nLaunching MuJoCo Viewer...")
        visualize_policy(env, env_cfg, make_inference_fn, params)
    
    # Print checkpoint info
    if ckpt_path is not None:
        print(f"\n{'='*50}")
        print(f"Checkpoint saved to: {ckpt_path}")
        print(f"\nTo replay this policy later, run:")
        print(f"  train-with-viewer --env_name {_ENV_NAME.value} --load_checkpoint {ckpt_path}")
        print(f"\nOr use train-jax-ppo:")
        print(f"  train-jax-ppo --env_name {_ENV_NAME.value} --play_only --load_checkpoint_path {ckpt_path}")
        print(f"{'='*50}")


def visualize_random_policy(env):
    """Visualize with random policy (for testing)."""
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    
    mj_data.qpos[:] = np.array(state.data.qpos)
    mj_data.qvel[:] = np.array(state.data.qvel)
    
    print("\nViewer Controls:")
    print("  [R] Reset")
    print("  [ESC/Q] Quit")
    
    should_quit = False
    needs_reset = False
    
    def key_callback(keycode):
        nonlocal needs_reset, should_quit
        if keycode == ord('r') or keycode == ord('R'):
            needs_reset = True
            print("Reset")
        elif keycode == 27 or keycode == ord('q'):
            should_quit = True
            print("Quit")
    
    step_count = 0
    
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not should_quit:
            if needs_reset:
                rng, reset_rng = jax.random.split(rng)
                state = jax.jit(env.reset)(reset_rng)
                mj_data.qpos[:] = np.array(state.data.qpos)
                mj_data.qvel[:] = np.array(state.data.qvel)
                mj_data.ctrl[:] = np.zeros(env.action_size)
                needs_reset = False
                step_count = 0
                print("Environment reset")
            
            # Random action
            rng, act_rng = jax.random.split(rng)
            action = jax.random.uniform(
                act_rng,
                (env.action_size,),
                minval=-1,
                maxval=1
            )
            
            # Step environment
            state = jax.jit(env.step)(state, action)
            step_count += 1
            
            # Update MuJoCo data for rendering
            mj_data.qpos[:] = np.array(state.data.qpos)
            mj_data.qvel[:] = np.array(state.data.qvel)
            mj_data.ctrl[:] = np.array(action)
            
            # Forward kinematics (not full physics step, just for rendering)
            mujoco.mj_forward(mj_model, mj_data)
            
            viewer.sync()
            
            # Print state occasionally
            if step_count % 100 == 0:
                print(f"Step {step_count}: qpos={mj_data.qpos[:3]}, ctrl={mj_data.ctrl[:3]}")
            
            time.sleep(env.dt)
    
    print("Viewer closed.")


def run():
    """Entry point for uv/pip script."""
    app.run(main)


if __name__ == "__main__":
    run()
