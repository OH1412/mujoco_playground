#!/usr/bin/env python3
"""示例训练/调试脚本。

此文件不包含实际环境定义，只用于存放查看/训练逻辑，
并可以通过XML路径加载机器人描述。真实任务实现应在
其他位置完成，之后可以将环境类重新注册到库中。

用法：
    # 只查看仿真效果（不训练）
    python learning/train_spirobot.py --play_only --num_videos 5

    # 简短训练后看效果
    python learning/train_spirobot.py --num_timesteps 100000 --num_envs 256

    # 更长的训练
    python learning/train_spirobot.py --num_timesteps 5000000 --num_envs 2048
"""
import datetime
import functools
import json
import os
import time
import warnings
from pathlib import Path

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
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
import tensorboardX

try:
    import wandb
except ImportError:
    wandb = None


# 环境设置
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


# 命令行参数
_SEED = flags.DEFINE_integer("seed", 1, "随机种子")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 500_000, "训练步数"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 3, "生成多少个视频"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "评估次数")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "奖励缩放")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 200, "Episode 长度")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "观察值标准化"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "动作重复次数")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "展开长度")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Minibatch 数"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "每个 batch 的更新次数"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "折扣因子")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "学习率")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "熵成本")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 512, "并行环境数")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 64, "评估环境数"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch 大小")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "最大梯度范数")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "PPO 裁剪参数"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [128, 128],
    "策略网络隐层大小",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [256, 256],
    "值网络隐层大小",
)
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "仅推理，不训练"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", False, "使用 Weights & Biases 日志"
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", True, "使用 TensorBoard 日志"
)


def create_spirobot_env():
    """占位函数。原环境已从库中移除。

    本脚本仅保存可视化/训练逻辑；实际的 `Spirobot` 环境应
    在其他模块中实现并注册到 `mujoco_playground.registry`。
    """
    print("⚠️ create_spirobot_env(): 环境不在此处定义，返回 None。")
    return None


def main(argv):
    """运行 Spirobot 训练/推理。"""
    del argv
    
    print("=" * 60)
    print("🤖 MuJoCo Playground - Spirobot 强化学习演示")
    print("=" * 60)
    
    # 尝试创建环境
    try:
        env = create_spirobot_env()
        if env is None:
            print("\n❌ 没有可用环境，脚本结束。")
            return
    except Exception as e:
        print(f"\n❌ 环境创建失败: {e}")
        print("\n💡 可能的解决方案:")
        print("   1. 检查 XML 文件是否存在")
        print("   2. 检查 meshes 文件夹中的 STL 文件是否完整")
        print("   3. 运行以下命令检查其他可用环境:")
        print(f"      python -c \"from mujoco_playground import registry; print(registry.ALL_ENVS[:10])\"")
        return
    
    # 生成实验名称
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"Spirobot-{timestamp}"
    
    if _PLAY_ONLY.value:
        exp_name += "-play-only"
    
    print(f"\n📝 实验名称: {exp_name}")
    
    # 设置日志目录
    logdir = epath.Path("logs").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"📁 日志目录: {logdir}")
    
    # 初始化 TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
        writer = tensorboardX.SummaryWriter(logdir)
    
    # 初始化 WandB
    if _USE_WANDB.value and not _PLAY_ONLY.value:
        if wandb is None:
            print("⚠️  wandb 未安装，跳过 W&B 日志")
        else:
            try:
                wandb.init(project="mujoco-playground", name=exp_name)
            except Exception as e:
                print(f"⚠️  WandB 初始化失败: {e}")
    
    # 创建网络配置
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=tuple(map(int, _POLICY_HIDDEN_LAYER_SIZES.value)),
        value_hidden_layer_sizes=tuple(map(int, _VALUE_HIDDEN_LAYER_SIZES.value)),
    )
    
    # 环境包装
    env = wrapper.wrap_for_brax_training(
        env,
        episode_length=_EPISODE_LENGTH.value,
        action_repeat=_ACTION_REPEAT.value,
    )
    
    print(f"\n✅ 环境准备完成")
    print(f"   观察空间大小: {env.observation_size}")
    print(f"   动作空间大小: {env.action_size}")
    print(f"   Episode 长度: {_EPISODE_LENGTH.value}")
    
    # 设置训练参数
    # base params
    training_params = dict(
        num_timesteps=int(_NUM_TIMESTEPS.value),
        num_evals=int(_NUM_EVALS.value),
        reward_scaling=float(_REWARD_SCALING.value),
        episode_length=int(_EPISODE_LENGTH.value),
        normalize_observations=_NORMALIZE_OBSERVATIONS.value,
        action_repeat=int(_ACTION_REPEAT.value),
        unroll_length=int(_UNROLL_LENGTH.value),
        num_minibatches=int(_NUM_MINIBATCHES.value),
        num_updates_per_batch=int(_NUM_UPDATES_PER_BATCH.value),
        num_envs=int(_NUM_ENVS.value),
        batch_size=int(_BATCH_SIZE.value),
        seed=_SEED.value,
        learning_rate=float(_LEARNING_RATE.value),
        entropy_cost=float(_ENTROPY_COST.value),
        discounting=float(_DISCOUNTING.value),
        max_grad_norm=float(_MAX_GRAD_NORM.value),
        clipping_epsilon=float(_CLIPPING_EPSILON.value),
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=int(_NUM_EVAL_ENVS.value),
    )
    # adjust for play-only mode to avoid batching issues
    if _PLAY_ONLY.value:
        training_params["num_timesteps"] = 0
        training_params["num_envs"] = 1
        training_params["num_eval_envs"] = 1
    
    print(f"\n⚙️  训练参数:")
    print(f"   总步数: {_NUM_TIMESTEPS.value:,}")
    print(f"   并行环境: {_NUM_ENVS.value}")
    print(f"   学习率: {_LEARNING_RATE.value}")
    print(f"   Batch 大小: {_BATCH_SIZE.value}")
    
    times = [time.monotonic()]
    
    def progress(num_steps, metrics):
        """训练进度回调。"""
        times.append(time.monotonic())
        
        if _USE_TB.value and not _PLAY_ONLY.value:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, num_steps)
            writer.flush()
        
        if _USE_WANDB.value and not _PLAY_ONLY.value:
            try:
                wandb.log(metrics, step=num_steps)
            except:
                pass
        
        if num_steps % 10000 == 0:
            if "eval/episode_reward" in metrics:
                print(f"   步数 {num_steps:7d}: 奖励 = {metrics['eval/episode_reward']:.3f}")
    
    # 开始训练/推理
    print(f"\n🚀 启动 {'推理' if _PLAY_ONLY.value else '训练'}...")
    
    try:
        train_fn = functools.partial(
            ppo.train,
            **training_params,
        )
        
        make_inference_fn, params, _ = train_fn(
            environment=env,
            progress_fn=progress,
        )
        
        print(f"\n✅ {'推理' if _PLAY_ONLY.value else '训练'} 完成！")
        
        if len(times) > 1:
            print(f"\n⏱️  时间统计:")
            print(f"   JIT 编译时间: {times[1] - times[0]:.2f} 秒")
            print(f"   {'推理' if _PLAY_ONLY.value else '训练'}时间: {times[-1] - times[1]:.2f} 秒")
        
        # 测试推理
        print(f"\n🎬 生成 {_NUM_VIDEOS.value} 个演示视频...")
        
        eval_env = registry.load("CartpoleBalance")  # 用于视频渲染
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)
        
        print(f"✅ 视频生成流程完成")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    app.run(main)
