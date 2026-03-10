# Copyright 2026
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
"""示例环境：Spirobot 抓取矿泉水瓶并放到目标位置。

这个文件是一份模板，你可以根据具体任务进一步完善，再将环境
注册到库的 registry 中或者在自己的模块里重写。
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
import xml
import xml.etree.ElementTree as ET

from mujoco_playground._src import mjx_env


def quat_rotate(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector(s) *v* by quaternion *q*.

    Formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
    Works with broadcasting over leading dimensions of *v*.
    """
    # q = [w, x, y, z]
    w = q[0]
    xyz = q[1:]
    return v + 2.0 * jp.cross(xyz, jp.cross(xyz, v) + w * v)


def default_config() -> config_dict.ConfigDict:
    """返回默认配置，用于训练脚本。

    可以在外部使用 `registry.get_default_config("SpirobotBottle")` 获得
    这个配置并据此覆盖键值。
    """
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=500,
        action_repeat=1,
        vision=False,
        impl="warp",
        # 训练任务相关的位置：初始和目标 XY 相对值（单位米）
        bottle_start=jp.array([-0.5, 0.0]),
        bottle_goal=jp.array([0.5, 0.0]),
        # 每次 reset 时随机决定机械臂应悬停的圆周半径（米）
        hover_radius=0.1,
        # 是否在场景中添加地面平面（便于调试/避免机器人掉落）
        add_ground=True,
    )


def domain_randomize(model: mjx.Model, rng: jax.Array) -> mjx.Model:
    """域随机化示例，暂时不做任何改动。"""
    return model


class SpirobotBottle(mjx_env.MjxEnv):
    """示例环境实现。

    机器人拥有一个机械臂 + spirobot 手爪，在空间内搜索一个矿泉水瓶，
    将其抓起并放置到预先指定位置。
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)
        # 基本机器人模型
        base_xml = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "spirobot"
            / "xmls"
            / "spirobot"
            / "ROBOT_ball_joint.xml"
        )
        # 保存原始 xml 路径以满足抽象接口
        self._xml_path = base_xml
        # 瓶子模型现在存放在 xmls 目录同级的 bottle_cap 文件夹
        bottle_xml = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "spirobot"
            / "xmls"
            / "bottle_cap"
            / "bottle_cap.xml"
        )
        # 合并瓶子 XML：直接将其 worldbody 内容拷贝进 base 的 worldbody
        base_str = base_xml.read_text()
        root = xml.etree.ElementTree.fromstring(base_str)
        # 如果需要，给场景添加简单地面和平面材质
        if self._config.add_ground:
            wb = root.find("worldbody")
            if wb is None:
                wb = xml.etree.ElementTree.SubElement(root, "worldbody")
            # 增加一个靠近 z=0 的平面 geom
            xml.etree.ElementTree.SubElement(
                wb,
                "geom",
                {
                    "name": "floor",
                    "type": "plane",
                    "pos": "0 0 0",
                    "size": "2 2 0.1",
                    "material": "grid",
                },
            )
            asset = root.find("asset")
            if asset is None:
                asset = xml.etree.ElementTree.SubElement(root, "asset")
            # checker 纹理 + 材质
            tex = xml.etree.ElementTree.SubElement(asset, "texture")
            tex.set("name", "grid")
            tex.set("type", "2d")
            tex.set("builtin", "checker")
            tex.set("width", "300")
            tex.set("height", "300")
            tex.set("mark", "edge")
            tex.set("markrgb", ".3 .3 .3")
            mat = xml.etree.ElementTree.SubElement(asset, "material")
            mat.set("name", "grid")
            mat.set("texture", "grid")
            mat.set("texrepeat", "1 1")
            mat.set("texuniform", "true")
            mat.set("reflectance", ".2")
        # ensure base mesh/texture paths are absolute so MuJoCo can open them
        base_comp = root.find("compiler")
        base_meshdir = None
        if base_comp is not None and base_comp.get("meshdir"):
            base_meshdir = base_comp.get("meshdir")
        base_asset = root.find("asset")
        if base_asset is not None:
            for child in list(base_asset):
                if child.tag == "mesh" and child.get("file"):
                    orig = child.get("file")
                    if base_meshdir:
                        newpath = base_xml.parent / base_meshdir / orig
                    else:
                        newpath = base_xml.parent / orig
                    child.set("file", str(newpath))
                if child.tag == "texture" and child.get("file"):
                    tex = child.get("file")
                    child.set("file", str(base_xml.parent / tex))
        wb = root.find("worldbody")
        if wb is None:
            wb = xml.etree.ElementTree.SubElement(root, "worldbody")
        # 读取瓶子 XML 并合并资产和 worldbody
        bottle_str = bottle_xml.read_text()
        bottle_root = xml.etree.ElementTree.fromstring(bottle_str)
        bottle_asset = bottle_root.find("asset")
        base_asset = root.find("asset")
        # get meshdir from bottle compiler if any
        bottle_meshdir = None
        comp = bottle_root.find("compiler")
        if comp is not None and comp.get("meshdir"):
            bottle_meshdir = comp.get("meshdir")
        if bottle_asset is not None:
            if base_asset is None:
                base_asset = xml.etree.ElementTree.SubElement(root, "asset")
            for child in list(bottle_asset):
                if child.tag == "mesh" and child.get("file"):
                    orig = child.get("file")
                    if bottle_meshdir:
                        newpath = bottle_xml.parent / bottle_meshdir / orig
                    else:
                        newpath = bottle_xml.parent / orig
                    child.set("file", str(newpath))
                if child.tag == "texture" and child.get("file"):
                    tex = child.get("file")
                    child.set("file", str(bottle_xml.parent / tex))
                base_asset.append(child)
        bottle_wb = bottle_root.find("worldbody")
        if bottle_wb is not None:
            for child in list(bottle_wb):
                wb.append(child)
        merged_str = xml.etree.ElementTree.tostring(root, encoding="unicode")
        # 创建模型
        self._mj_model = mujoco.MjModel.from_xml_string(merged_str)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._post_init()

    def _post_init(self) -> None:
        # 关节列表
        # 只选取机械臂关节；软体球关节名为 “joint_ball_*”，应排除
        self._arm_joint_ids = []
        self._ball_joint_ids = []  # 单独记录球关节，以备调试或分析
        self._ball_qpos_adrs = []  # 记录每个球关节在 qpos 中的起始 adr
        for i in range(self._mj_model.njnt):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not name:
                continue
            if name.startswith("joint_ball_"):
                self._ball_joint_ids.append(i)
                # 球关节在 qpos 中占 4 个元素，起始地址由 qposadr
                # mj_model.joint(...).qposadr is a numpy array scalar
                adr = int(self._mj_model.joint(name).qposadr.item())
                self._ball_qpos_adrs.append(adr)
            # 期望格式 joint_1 … joint_6 等普通转动关节
            # 排除以 joint_ball_ 开头的球关节
            elif name.startswith("joint_"):
                self._arm_joint_ids.append(i)

        # 记录末端执行器刚体 site 与瓶子 body，用于 obs/reward
        self._rigid_ee_site_id = self._mj_model.site("rigid_ee_site").id
        self._bottle_body_id = self._mj_model.body("base").id
        # 记录触手各段 body id，用于计算累计卷绕角
        self._finger_body_ids = []
        for i in range(1, 20):
            name = f"finger_seg{i}"
            try:
                bid = self._mj_model.body(name).id
            except Exception:
                continue  # 若某段不存在则跳过
            self._finger_body_ids.append(bid)
        # convert to fixed numpy array for reliable JAX indexing
        self._finger_body_ids = np.array(self._finger_body_ids, dtype=np.int32)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, sub = jax.random.split(rng)
        qpos = jp.zeros(self._mjx_model.nq)
        qvel = jp.zeros(self._mjx_model.nv)
        ctrl = jp.zeros(self._mjx_model.nu)
        # 如果有自由关节, 用配置和 xml 初始化 xy 及姿态
        # （假设模型一定包含 base_free 和 body base）
        jid = self._mj_model.joint("base_free").id
        adr = int(self._mj_model.joint(jid).qposadr.item())
        bid = self._mj_model.body("base").id
        base_z = jp.array(self._mj_model.body(bid).pos)[2]
        base_quat = jp.array(self._mj_model.body(bid).quat)
        # 添加随机扰动
        rng, r1, r2, r3 = jax.random.split(rng, 4)
        noise = jax.random.uniform(r1, (2,), minval=-0.05, maxval=0.05)
        start_xy = self._config.bottle_start + noise
        theta = jax.random.uniform(r2, minval=-jp.pi, maxval=jp.pi)
        rand_quat = jp.array([jp.cos(theta/2), 0.0, 0.0, jp.sin(theta/2)])
        qpos = qpos.at[adr:adr+2].set(start_xy)
        qpos = qpos.at[adr+2].set(base_z)
        qpos = qpos.at[adr+3:adr+7].set(rand_quat)
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            impl=self._mjx_model.impl.value,
        )
        data = mjx.forward(self._mjx_model, data)
        # 在 reset 时随机选择一个 hover 方向用于悬停点
        angle = jax.random.uniform(r3, minval=0.0, maxval=2 * jp.pi)
        hover_offset = jp.array([jp.cos(angle), jp.sin(angle), 0.0]) * self._config.hover_radius
        info = {"rng": rng, "step_count": jp.array(0), "hover_offset": hover_offset}
        obs = self._get_obs(data, info, hover_offset)
        metrics = {}
        # 初始化球关节为单位四元数避免非法状态
        for adr in self._ball_qpos_adrs:
            qpos = qpos.at[adr].set(1.0)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        data = mjx_env.step(self._mjx_model, state.data, action, self.n_substeps)
        # 不要修改 data 对象，信息从 state.info 传入
        obs = self._get_obs(data, state.info)
        reward = self._compute_reward(data, state.info)
        done = self._check_done(data, state.info)
        info = state.info
        info["step_count"] = info.get("step_count", jp.array(0)) + 1
        # attach posture metrics for logging/inspection
        # note: reward already includes r_posture
        # we can recompute tilt_cos & r_posture here if desired
        return mjx_env.State(data, obs, reward, done, {}, info)

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any], hover_offset: Optional[jax.Array] = None) -> jax.Array:
        # 本体关节位置/速度
        qpos = data.qpos
        qvel = data.qvel
        # 绝对空间位置和线/角速度
        ee_pos = data.site_xpos[self._rigid_ee_site_id]
        # warp backend may not provide site velocities
        if hasattr(data, "site_xvelp"):
            ee_vel = data.site_xvelp[self._rigid_ee_site_id]
        else:
            ee_vel = jp.zeros(3)
        bottle_pos = data.xpos[self._bottle_body_id]
        # some backends omit linear/ang vel arrays
        bottle_vel = data.xvelp[self._bottle_body_id] if hasattr(data, "xvelp") else jp.zeros(3)
        bottle_angvel = data.xvelr[self._bottle_body_id] if hasattr(data, "xvelr") else jp.zeros(3)
        bottle_quat = data.xquat[self._bottle_body_id]
        goal_pos = jp.array([self._config.bottle_goal[0], self._config.bottle_goal[1], 0.0])
        # 如果没有外部传入 hover_offset（例如执行 step），从 info 中取
        # 注：网络无需再看到 3D 的 hover_target；我们改为给出水平表面距离误差
        if hover_offset is None:
            hover_offset = info.get("hover_offset", jp.zeros(3))
        # 计算到瓶子圆柱表面的水平偏差向量（z=0）
        xy_rel = ee_pos[:2] - bottle_pos[:2]
        dist_xy = jp.linalg.norm(xy_rel)
        # 避免除以0
        dir_xy = xy_rel / (dist_xy + 1e-8)
        bottle_radius = 0.03
        surface_xy = bottle_pos[:2] + dir_xy * bottle_radius
        ee_to_hover = jp.concatenate([surface_xy - ee_pos[:2], jp.array([0.0])])
        ee_to_bottle = bottle_pos - ee_pos
        bottle_to_goal = goal_pos - bottle_pos
        # 累计卷绕角：使用与奖励相同的局部坐标系转换
        finger_world_pos = data.xpos[self._finger_body_ids]
        rel_pos = finger_world_pos - bottle_pos
        bottle_quat_inv = jp.array([bottle_quat[0], -bottle_quat[1], -bottle_quat[2], -bottle_quat[3]])
        local_pos = quat_rotate(bottle_quat_inv, rel_pos)
        local_xz = local_pos[:, [0, 2]]
        
        # 【修改1.1：添加闭合操作】
        local_xz_closed = jp.concatenate([local_xz, local_xz[0:1]], axis=0) 
        thetas = jp.arctan2(local_xz_closed[:, 1], local_xz_closed[:, 0])
        dtheta = thetas[1:] - thetas[:-1]
        dtheta = (dtheta + jp.pi) % (2 * jp.pi) - jp.pi
        
        # 【修改1.2：分别算出闭合和开口角度】
        closed_theta_total = jp.sum(dtheta)
        open_theta_total = jp.abs(jp.sum(dtheta[:-1]))
        
        # 把动作也放进去可以选用（有助于策略学习）
        ctrl = data.ctrl
        return jp.concatenate([
            qpos, qvel, ctrl,
            ee_pos, ee_vel,
            bottle_pos, bottle_vel, bottle_angvel, bottle_quat,
            goal_pos,
            ee_to_bottle, ee_to_hover, bottle_to_goal,
            # 【修改1.3：把两个核心状态传给网络】
            jp.array([closed_theta_total, open_theta_total]),
        ])

    def _compute_reward(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        # 计算所有奖励组件，包括悬停靠近、包裹、抬起、放置/成功、
        # 以及新增的姿态保持奖励（避免瓶子倾倒）。
        ee_pos = data.site_xpos[self._rigid_ee_site_id]
        bottle_pos = data.xpos[self._bottle_body_id]
        goal_pos = jp.array([self._config.bottle_goal[0], self._config.bottle_goal[1], 0.0])
        # 计算 hover_target：从 info 读取悬停偏移量
        hover_offset = info.get("hover_offset", jp.zeros(3))
        hover_target = bottle_pos + hover_offset
        bottle_quat = data.xquat[self._bottle_body_id]  # make quaternion available
        d_ee_hover = jp.linalg.norm(ee_pos - hover_target)
        # 计算圆柱表面最短距离
        rel_ee_pos = ee_pos - bottle_pos
        bottle_quat_inv = jp.array([bottle_quat[0], -bottle_quat[1], -bottle_quat[2], -bottle_quat[3]])
        ee_local_pos = quat_rotate(bottle_quat_inv, rel_ee_pos)
        ee_local_y = ee_local_pos[1]
        ee_local_xz = jp.linalg.norm(ee_local_pos[[0, 2]])
        bottle_radius = 0.03
        bottle_half_height = 0.10
        dy = jp.maximum(0.0, jp.abs(ee_local_y) - bottle_half_height)
        dxz = jp.maximum(0.0, ee_local_xz - bottle_radius)
        d_ee_surface = jp.sqrt(dxz * dxz + dy * dy)
        r_reach = jp.exp(-10.0 * d_ee_surface)
        # 宏观靠近判定：贴近表面且不靠近顶底
        is_macro_close = (d_ee_surface < 0.15) & (dy < 0.05)
        # 2. 计算触手在瓶子本地坐标系中的位置并绕柱面算角度
        finger_world_pos = data.xpos[self._finger_body_ids]
        rel_pos = finger_world_pos - bottle_pos
        # 逆四元数：旋转世界 → 本地
        bottle_quat_inv = jp.array([bottle_quat[0], -bottle_quat[1], -bottle_quat[2], -bottle_quat[3]])
        local_pos = quat_rotate(bottle_quat_inv, rel_pos)

        # 3. 在本地 XZ 平面上计算累计卷绕角（瓶轴在本地 Y）
        # use plain Python list for static slicing, avoids tracer issues
        local_xz = local_pos[:, [0, 2]]
        # 【修改2.1：补回首尾相连的防作弊闭合逻辑】
        local_xz_closed = jp.concatenate([local_xz, local_xz[0:1]], axis=0) 
        thetas = jp.arctan2(local_xz_closed[:, 1], local_xz_closed[:, 0])
        dtheta = thetas[1:] - thetas[:-1]
        dtheta = (dtheta + jp.pi) % (2.0 * jp.pi) - jp.pi
        # 【修改2.2：定义好 closed 和 open 两个变量，防止 NameError】
        closed_theta_total = jp.sum(dtheta)
        open_theta_total = jp.abs(jp.sum(dtheta[:-1]))

        # 4. 严苛的包裹合法性约束
        is_height_valid = jp.mean(jp.abs(local_pos[:, 1])) < 0.1
        local_radii = jp.linalg.norm(local_xz, axis=-1)
        is_radius_valid = jp.mean(jp.abs(local_radii - 0.03)) < 0.05
        # 【修改2.3：加上严格的拓扑判断，确保瓶子在中间】
        is_inside = jp.abs(closed_theta_total) > 3.0
        # 【修改2.4：将 is_inside 纳入真理之门判定】
        is_valid_wrap = is_macro_close & is_height_valid & is_radius_valid & is_inside

        # wrap 同 debug_env：r_wrap_dense 只要求 macro_close & height_valid
        r_wrap_dense = jp.where(is_macro_close & is_height_valid, open_theta_total * 1.0, 0.0)
        r_wrap_jackpot = jp.where(is_valid_wrap, 20.0, 0.0)
        r_wrap = r_wrap_dense + r_wrap_jackpot

        # 抬起状态：基于初始高度 + 0.05
        init_z = info.get("init_bottle_z", 0.0)
        is_lifted = bottle_pos[2] > (init_z + 0.05)
        # 抬起奖励现在需要同时满足 wrap 有效和已离地
        r_lift = jp.where(is_valid_wrap & is_lifted, 1.0, 0.0)
        d_bottle_goal = jp.linalg.norm(bottle_pos - goal_pos)
        # 放置和成功只依赖于 is_lifted
        r_place = jp.where(is_lifted, jp.exp(-10.0 * d_bottle_goal), 0.0)
        r_success = jp.where(d_bottle_goal < 0.05, 5.0, 0.0)
        # ---------------- 3.5 姿态保持奖励 ----------------
        # 计算局部 Y 轴在世界 Z 轴的投影
        qw, qx, qy, qz = bottle_quat[0], bottle_quat[1], bottle_quat[2], bottle_quat[3]
        tilt_cos = jp.clip(2.0 * (qw * qx + qy * qz), -1.0, 1.0)
        r_posture = jp.where(is_macro_close, jp.exp(-3.0 * (1.0 - tilt_cos)), 0.0)
        # 动作惩罚区分前 6 个刚性电机和后 3 个线驱动
        ctrl = data.ctrl
        r_penalty = 0.01 * jp.sum(jp.square(ctrl[:6])) + 1.0 * jp.sum(jp.square(ctrl[6:]))
        return r_reach + r_wrap + r_lift + r_place + r_success + r_posture - r_penalty
    def _check_done(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        # jax-friendly boolean operations only
        # timeout based on step_count rather than realtime seconds
        timeout = info.get("step_count", jp.array(0)) >= self._config.episode_length
        bottle_pos = data.xpos[self._bottle_body_id]
        goal_pos = jp.array([self._config.bottle_goal[0], self._config.bottle_goal[1], 0.0])
        d_bottle_goal = jp.linalg.norm(bottle_pos - goal_pos)
        # success when near goal and near initial height
        init_z = info.get("init_bottle_z", jp.array(0.0))
        success = (d_bottle_goal < 0.05) & (jp.abs(bottle_pos[2] - init_z) < 0.02)
        # 使用初始高度减去0.08作为跌落阈值
        init_z = info.get("init_bottle_z", jp.array(0.0))
        drop_threshold = init_z - 0.08
        dropped = bottle_pos[2] < drop_threshold          # 瓶子跌出“桌面”
        fallen = (bottle_pos[2] < drop_threshold) & (~success)  # 倒地失败
        # nan check
        nan_fail = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = timeout | success | dropped | fallen | nan_fail
        return done.astype(jp.float32)

    @property
    def xml_path(self) -> str:
        # 抽象基类要求提供 xml 路径，可用于记录或调试
        return str(self._xml_path)

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
