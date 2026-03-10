import os
# 【极其关键】：在导入 JAX 之前彻底禁用 GPU，强制使用 CPU，防止显卡崩溃！
os.environ['JAX_PLATFORMS'] = 'cpu' 

import time
import mujoco
import mujoco.viewer
import numpy as np

# 导入你的环境类
from mujoco_playground._src.manipulation.spirobot.spirobot_bottle import (
    SpirobotBottle,
    default_config,
)

# ==============================================================================
# 纯 NumPy 版本的数学辅助函数
# ==============================================================================
def quat_rotate_numpy(q, v):
    """纯 NumPy 版本的四元数旋转"""
    w = q[0]
    xyz = q[1:]
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (w * uv + uuv)

def compute_reward_numpy(mj_data, env, info):
    """
    纯 NumPy 的奖励计算函数（包含所有奖励项）。
    """
    ee_pos = mj_data.site_xpos[env._rigid_ee_site_id].copy()
    bottle_pos = mj_data.xpos[env._bottle_body_id].copy()
    bottle_quat = mj_data.xquat[env._bottle_body_id].copy()
    goal_pos = np.array([env._config.bottle_goal[0], env._config.bottle_goal[1], 0.0])
    
    # ---------------- 1. 悬停靠近 (Reach) ----------------
    hover_offset = info.get("hover_offset", np.zeros(3))
    hover_target = bottle_pos + hover_offset
    d_ee_hover = float(np.linalg.norm(ee_pos - hover_target))
    # 计算水平方向与瓶子中心距离，减去瓶半径得到表面距离
    d_xy_center = float(np.linalg.norm((ee_pos - bottle_pos)[:2]))
    bottle_radius = 0.03
    d_xy_surface = max(0.0, d_xy_center - bottle_radius)
    # 修改宏观靠近判定为水平表面距离小于 0.20
    is_macro_close = d_xy_surface < 0.20
    # 使用表面距离而非 hover 误差引导策略
    r_reach = np.exp(-10.0 * d_xy_surface)
    
    # ---------------- 2. 局部卷绕与包裹 (Wrap) ----------------
    finger_ids = np.array(env._finger_body_ids)
    finger_world_pos = mj_data.xpos[finger_ids].copy()
    rel_pos = finger_world_pos - bottle_pos
    
    bottle_quat_inv = np.array([bottle_quat[0], -bottle_quat[1], -bottle_quat[2], -bottle_quat[3]])
    local_pos = quat_rotate_numpy(bottle_quat_inv, rel_pos)
    
    local_xz = local_pos[:, [0, 2]]
    local_xz_closed = np.concatenate([local_xz, local_xz[0:1]], axis=0) 
    
    thetas = np.arctan2(local_xz_closed[:, 1], local_xz_closed[:, 0])
    dtheta = thetas[1:] - thetas[:-1]
    dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi
    
    closed_theta_total = float(np.sum(dtheta))
    open_theta_total = float(np.abs(np.sum(dtheta[:-1])))
    
    # 约束条件
    # 采用之前计算的表面距离
    height_mean = float(np.mean(np.abs(local_pos[:, 1])))
    is_height_valid = bool(height_mean < 0.1)
    local_radii = np.linalg.norm(local_xz, axis=-1)
    radius_diff = float(np.mean(np.abs(local_radii - 0.03)))
    # 改为 5cm 容许度
    is_radius_valid = bool(radius_diff < 0.05)
    is_inside = bool(np.abs(closed_theta_total) > 3.0)
    
    is_valid_wrap = is_macro_close and is_height_valid and is_radius_valid and is_inside
    r_wrap = (open_theta_total * 1.0 if (is_macro_close and is_height_valid) else 0.0) + \
             (20.0 if is_valid_wrap else 0.0)
    
    # ---------------- 3. 抬起、放置与成功 (Lift, Place, Success) ----------------
    d_ee_bottle = float(np.linalg.norm(bottle_pos - ee_pos))
    bottle_z = float(bottle_pos[2])
    d_bottle_goal = float(np.linalg.norm(bottle_pos - goal_pos))
    
    # 抬起判定基于初始高度加阈值0.05
    init_z = info.get("init_bottle_z", 0.0)
    is_lifted = bool(bottle_z > init_z + 0.05)
    
    # 只有包住并抬起来才给 lift 分
    r_lift = 1.0 if (is_valid_wrap and is_lifted) else 0.0
    r_place = np.exp(-10.0 * d_bottle_goal) if is_lifted else 0.0
    r_success = 5.0 if d_bottle_goal < 0.05 else 0.0
    
    # ---------------- 3.5 姿态保持加分 (Anti-Topple Posture) ----------------
    # 提取四元数四个分量 (q = [w, x, y, z])
    qw, qx, qy, qz = bottle_quat[0], bottle_quat[1], bottle_quat[2], bottle_quat[3]
    # 计算局部 Y 轴相对于世界 Z 轴的投影
    tilt_cos = 2.0 * (qw * qx + qy * qz)
    tilt_cos = float(np.clip(tilt_cos, -1.0, 1.0))
    r_posture = float(np.exp(-3.0 * (1.0 - tilt_cos))) if is_macro_close else 0.0
    
    # ---------------- 4. 动作惩罚 (Penalty) ----------------
    ctrl = mj_data.ctrl.copy()
    p_arm = 0.01 * float(np.sum(np.square(ctrl[:6])))
    p_tendon = 1.0 * float(np.sum(np.square(ctrl[6:])))
    r_penalty = p_arm + p_tendon
    
    # 总分结算
    total_reward = r_reach + r_wrap + r_lift + r_place + r_success + r_posture - r_penalty
    
    return {
        # 判定状态
        "d_ee_hover": d_ee_hover, "d_xy_surface": d_xy_surface, "is_macro_close": is_macro_close,
        "height_mean": height_mean, "is_height_valid": is_height_valid,
        "radius_diff": radius_diff, "is_radius_valid": is_radius_valid,
        "closed_theta_total": closed_theta_total, "is_inside": is_inside,
        "d_ee_bottle": d_ee_bottle,
        "bottle_z": bottle_z, "is_lifted": is_lifted,
        "d_bottle_goal": d_bottle_goal,
        # 姿态相关
        "tilt_cos": tilt_cos, "r_posture": r_posture,
        # 奖励数值
        "r_reach": r_reach, "r_wrap": r_wrap, "r_lift": r_lift, 
        "r_place": r_place, "r_success": r_success,
        "p_arm": p_arm, "p_tendon": p_tendon, "r_penalty": r_penalty,
        "total": total_reward
    }

# ==============================================================================
# 主交互逻辑
# ==============================================================================
def main():
    config = default_config()
    env = SpirobotBottle(config=config)
    mj_model = env.mj_model
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    
    # 记录初始瓶子高度，后续抬起判定使用相对值
    init_bottle_z = float(mj_data.xpos[env._bottle_body_id][2])
    info = {"hover_offset": np.array([config.hover_radius, 0.0, 0.0]),
            "init_bottle_z": init_bottle_z}
    
    print("启动纯 CPU 交互式 Viewer...")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        last_print_time = time.time()
        last_render_time = time.time() 
        
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(mj_model, mj_data)
            
            # 60Hz 画面渲染
            if time.time() - last_render_time > 1.0 / 60.0:
                viewer.sync()
                last_render_time = time.time()
            
            # 2Hz 终端输出，不卡顿
            if time.time() - last_print_time > 0.5:
                res = compute_reward_numpy(mj_data, env, info)
                
                print("\n" + "="*55)
                print("【第一阶段：悬停与包裹判定】")
                print(f"🎯 靠近判定 (<0.20m 表面距离) : {'✅' if res['is_macro_close'] else '❌'}  (表面距离: {res['d_xy_surface']:.3f}m)")
                print(f"📏 高度判定 (<0.10m) : {'✅' if res['is_height_valid'] else '❌'}  (当前: {res['height_mean']:.3f}m)")
                print(f"⭕ 贴合判定 (<0.05m) : {'✅' if res['is_radius_valid'] else '❌'}  (当前: {res['radius_diff']:.3f}m)")
                print(f"🌀 闭合拓扑值: {res['closed_theta_total']:.2f} 弧度 => 瓶子在内部: {'✅' if res['is_inside'] else '❌'}")
                
                print("\n【第二阶段：放置判定】")
                # 抓取判定由第一阶段闭合/贴合决定，这里不显示
                print(f"🚀 抬起判定 (瓶身高度 Z > 0.05m): {'✅' if res['is_lifted'] else '❌'} (当前: {res['bottle_z']:.3f}m)")
                # 【新增这一行看倾斜度】
                print(f"📐 直立姿态 (1.0为完美直立): {res['tilt_cos']:.3f} => 姿态分: {res['r_posture']:.2f}")
                print(f"🎯 放置误差 (距离终点): {res['d_bottle_goal']:.3f}m")
                
                print("\n【💰 奖励分数实时监控】")
                print(f"➕ 靠近分 (Reach)   : {res['r_reach']:>6.2f}")
                print(f"➕ 包裹分 (Wrap)    : {res['r_wrap']:>6.2f}")
                print(f"➕ 抬起分 (Lift)    : {res['r_lift']:>6.2f}")
                print(f"➕ 放置分 (Place)   : {res['r_place']:>6.2f}")
                print(f"🌟 成功分 (Success) : {res['r_success']:>6.2f}")
                print(f"➖ 动作惩罚 (Penalty): -{res['r_penalty']:<5.2f} (手臂: {res['p_arm']:.2f}, 线驱: {res['p_tendon']:.2f})")
                print("-" * 55)
                print(f"💎 当前总奖励 (Total): {res['total']:>6.2f}")
                print("="*55)
                
                last_print_time = time.time()
            
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()