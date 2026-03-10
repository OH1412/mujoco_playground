# Spirobot Bottle Task Documentation

本文档介绍仓库中新添加的 **Spirobot 矿泉水瓶抓取任务**以及相关辅助脚本。

---

## 环境类：`SpirobotBottle`
文件路径：`mujoco_playground/_src/manipulation/spirobot/spirobot_bottle.py`

### 功能概述
- 基于 MuJoCo XML 扩展包含机械臂和一个矿泉水瓶模型。
- 提供 JAX 兼容的 obs/reward/done 实现，可用于 Brax/PPO 训练。
- 奖励由多个阶段组成：
  1. **Reach**：水平向瓶子表面靠近（使用 cylinder surface distance）
  2. **Wrap**：触手沿瓶壁卷绕，附加“宏观靠近”、“高度/半径有效性”和
     拓扑闭合约束。包含稠密＋jackpot 奖励。
  3. **Lift/Place/Success**：抬高后移动到目标，成功靠近给终极奖励。
  4. **Posture**：防倾倒奖励，仅惩罚瓶子倾斜而不惩罚自转。
  5. **Penalty**：电机动作消耗，线驱动更高惩罚。
- `reset` 时随机生成悬停方向，存入 `info`；obs 中改为水平表面误差向量。
- `_check_done` 包含超时、成功、跌落、NaN 等条件。

### 配置与扩展
默认配置可通过 `default_config()` 获取，支持随机悬停半径、是否添加地面等。
可在训练时通过 registry 或脚本覆盖。

---

## 调试工具：`debug_env.py`
路径：`mujoco_playground/_src/manipulation/debug_env.py`

该脚本提供一个**CPU 版本的可视化和奖励分析器**。

- 使用 NumPy 计算与环境完全一致的奖励项，便于在 MuJoCo viewer 中逐步测试。
- 输出文本界面展示各阶段判定、奖励分数、姿态指标等。
- 在不依赖 JAX 的情况下运行，可快速验证新逻辑。
- 运行：
  ```bash
  python -m mujoco_playground._src.manipulation.debug_env
  ```

---

## 训练脚本：`learning/train_spirobot.py`

训练/推理脚本封装了 Brax/PPO 训练管线。

主要功能：

1. 定义命令行旗标（训练步数、并行环境、网络结构、日志选项等）。
2. 加载环境（通过 `wrapper.wrap_for_brax_training`）。
3. 设置 TensorBoard/W&B 日志，可在回调中导出模型参数。
4. 调用 `ppo.train(...)` 执行训练，并返回 inference 函数。
5. 如 `--play_only`，直接进行推理并生成视频。

示例用法：
```bash
# 训练 50 万步
python learning/train_spirobot.py --num_timesteps 500000 --num_envs 256

# 仅查看效果
python learning/train_spirobot.py --play_only --num_videos 5
```

可通过修改 `progress` 回调实现中途 checkpoint 存储。

---

## 查看脚本：`learning/view_spirobot.py`

用于独立渲染 MuJoCo XML 模型，不依赖任何训练环境。

特性：
- 支持三种场景：`default`/`bright`/`dark`。
- 可选择仅显示机械臂、仅显示瓶子或两者合并。
- 自动在环境中添加地面、光源、天空盒；通过命令行参数控制。
- 提供简单周期运动示例，可用于快速检查碰撞和视觉。

用法示例：
```bash
python learning/view_spirobot.py --scene dark --duration 60
```

脚本主要用于开发阶段查看模型是否正常加载。

---

## 使用建议

1. 使用 `debug_env.py` 调试奖励逻辑，确认各项指标输出与预期一致。
2. 运行 `view_spirobot.py` 检查 XML 合并是否成功、网格位置是否对齐。
3. 修改 `train_spirobot.py` 的 `progress` 回调以便定期保存参数，中断后可继续训练。
4. 根据实验需要调整 `default_config()` 或在训练时传递 overrides。

以上内容构成了整个瓶子抓取任务的文档，便于后续团队成员理解设计及复现流程。