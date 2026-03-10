#!/usr/bin/env python3
"""增强版 Spirobot 查看器 - 仅用于视觉与调试。

本脚本不会依赖于任何注册环境；它直接加载 XML 并
在 MuJoCo 本地查看器中展示，便于检验模型和训练脚本。

用法：
    # 默认运行
    python3 learning/view_spirobot.py

    # 无地面（原始模型）
    python3 learning/view_spirobot.py --no_ground

    # 黑色场景（最小光源）
    python3 learning/view_spirobot.py --scene dark

    # 明亮场景
    python3 learning/view_spirobot.py --scene bright
"""

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np

# 设置环境变量以支持 GPU 渲染
os.environ["MUJOCO_GL"] = "egl"


def add_environment_to_xml(xml_path: Path, add_ground: bool = True, scene: str = "default") -> str:
    """为 XML 添加环境元素（地面、光源、背景等）。
    
    Args:
        xml_path: 原始 XML 文件路径
        add_ground: 是否添加地面
        scene: 场景类型 - 'default', 'bright', 'dark'
    
    Returns:
        修改后的 XML 字符串
    """
    
    # 读取原始 XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 1. 添加 asset（天空盒、材料等）
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    
    # 添加天空盒纹理（场景相关）
    if scene == "bright":
        rgb1, rgb2 = ".8 .9 1", ".5 .7 .9"
    elif scene == "dark":
        rgb1, rgb2 = ".1 .1 .15", "0 0 0"
    else:  # default
        rgb1, rgb2 = ".4 .6 .8", "0 0 0"
    
    skybox = ET.SubElement(asset, "texture")
    skybox.set("name", "skybox")
    skybox.set("type", "skybox")
    skybox.set("builtin", "gradient")
    skybox.set("rgb1", rgb1)
    skybox.set("rgb2", rgb2)
    skybox.set("width", "800")
    skybox.set("height", "800")
    
    # 添加地板纹理
    floor_tex = ET.SubElement(asset, "texture")
    floor_tex.set("name", "grid")
    floor_tex.set("type", "2d")
    floor_tex.set("builtin", "checker")
    
    if scene == "bright":
        floor_tex.set("rgb1", ".3 .4 .5")
        floor_tex.set("rgb2", ".5 .6 .7")
    elif scene == "dark":
        floor_tex.set("rgb1", ".05 .05 .08")
        floor_tex.set("rgb2", ".1 .1 .15")
    else:  # default
        floor_tex.set("rgb1", ".1 .2 .3")
        floor_tex.set("rgb2", ".2 .3 .4")
    
    floor_tex.set("width", "300")
    floor_tex.set("height", "300")
    floor_tex.set("mark", "edge")
    floor_tex.set("markrgb", ".3 .3 .3")
    
    # 添加材料
    grid_mat = ET.SubElement(asset, "material")
    grid_mat.set("name", "grid")
    grid_mat.set("texture", "grid")
    grid_mat.set("texrepeat", "1 1")
    grid_mat.set("texuniform", "true")
    grid_mat.set("reflectance", ".2")
    
    # 2. 添加 visual 设置
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    
    # 设置光源和渲染质量（场景相关）
    if visual.find("headlight") is None:
        headlight = ET.SubElement(visual, "headlight")
        if scene == "bright":
            headlight.set("ambient", ".6 .6 .6")
            headlight.set("diffuse", "1 1 1")
        elif scene == "dark":
            headlight.set("ambient", ".2 .2 .2")
            headlight.set("diffuse", ".4 .4 .4")
        else:  # default
            headlight.set("ambient", ".4 .4 .4")
            headlight.set("diffuse", ".8 .8 .8")
        headlight.set("specular", "0.1 0.1 0.1")
    
    if visual.find("map") is None:
        vmap = ET.SubElement(visual, "map")
        vmap.set("znear", ".01")
    
    if visual.find("quality") is None:
        quality = ET.SubElement(visual, "quality")
        quality.set("shadowsize", "2048")
    
    # 3. 在 worldbody 中添加地面和光源
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    
    # 添加光源（如果不存在）
    lights = worldbody.findall("light")
    if len(lights) == 0:
        light = ET.SubElement(worldbody, "light")
        light.set("name", "light")
        light.set("pos", "0 0 1")
        light.set("dir", "0 0 -1")
        
        if scene == "bright":
            light.set("intensity", "2")
        elif scene == "dark":
            light.set("intensity", "0.5")
    
    # 添加地板（如果需要且不存在）
    if add_ground:
        floors = worldbody.findall("geom[@name='floor']")
        if len(floors) == 0:
            floor = ET.SubElement(worldbody, "geom")
            floor.set("name", "floor")
            # 把地板放在 z=0，使底座靠地
            floor.set("pos", "0 0 0")
            floor.set("size", "2 2 0.1")
            floor.set("type", "plane")
            floor.set("material", "grid")
    
    # 转换回 XML 字符串
    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str


def main(args):
    """加载 Spirobot 并显示。"""
    
    # 定位 Spirobot XML 文件
    repo_root = Path(__file__).parent.parent
    base_path = repo_root / "mujoco_playground" / "_src" / "manipulation" / "spirobot" / "xmls" / "spirobot" / "ROBOT_ball_joint.xml"
    # 解析所需模型类型
    # 注意: bottle_cap 与 arm 模型分别位于不同路径
    bottle_path = base_path.parent.parent / "bottle_cap" / "bottle_cap.xml"
    if args.only == "bottle":
        # 只显示瓶子本身，绕过合并逻辑
        if not bottle_path.exists():
            print(f"❌ 找不到瓶子文件 {bottle_path}")
            return
        xml_path = bottle_path
    elif args.only == "arm":
        xml_path = base_path
    else:  # spirobot 模式
        if bottle_path.exists():
            # 居中偏移开关
            offset_enabled = not args.no_offset

            # 将瓶子模型内的 worldbody 元素移入基础 XML（避免 include 的 schema 冲突）
            import xml.etree.ElementTree as ET
            base_str = base_path.read_text()
            root = ET.fromstring(base_str)
            wb = root.find("worldbody")
            if wb is None:
                wb = ET.SubElement(root, "worldbody")

            # 读取瓶子 XML 并提取其 worldbody 及 asset 子元素
            bottle_root = ET.fromstring(bottle_path.read_text())

            # copy assets (mesh/texture/material)
            bottle_asset = bottle_root.find("asset")
            base_asset = root.find("asset")

            # 确定 bottle meshdir（如果存在）
            bottle_meshdir = None
            comp = bottle_root.find("compiler")
            if comp is not None and comp.get("meshdir"):
                bottle_meshdir = comp.get("meshdir")

            if bottle_asset is not None:
                if base_asset is None:
                    base_asset = ET.SubElement(root, "asset")
                for child in list(bottle_asset):
                    if child.tag == "mesh" and child.get("file"):
                        orig = child.get("file")
                        if bottle_meshdir:
                            newpath = bottle_path.parent / bottle_meshdir / orig
                        else:
                            newpath = bottle_path.parent / orig
                        child.set("file", str(newpath))
                    if child.tag == "texture" and child.get("file"):
                        tex = child.get("file")
                        child.set("file", str(bottle_path.parent / tex))
                    base_asset.append(child)

            # copy worldbody geometry
            bottle_wb = bottle_root.find("worldbody")
            if bottle_wb is not None:
                for child in list(bottle_wb):
                    if offset_enabled and child.tag == "body" and child.get("name") == "base":
                        if child.get("pos") is None:
                            child.set("pos", "0.2 0 0")
                    wb.append(child)

            # 将合并结果写到与 base_xml 同目录，以保留 mesh 相对路径
            xml_path = base_path.parent / "_temp_spirobot_view.xml"
            xml_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")

            # log asset count
            if base_asset is not None:
                print(f"[debug] merged assets count: {len(base_asset)}")
            # debug 输出合并后 body 名称列表
            try:
                m = mujoco.MjModel.from_xml_path(str(xml_path))
                names = [m.body(i).name for i in range(m.nbody)]
                print("[debug] merged model bodies:", names)
            except Exception as e:
                print("[debug] merge failed:", e)
        # 这里的 else 对应上面的 `if bottle_path.exists():`
        else:
            xml_path = base_path
    
    if not xml_path.exists():
        print(f"❌ 错误：XML 文件不存在: {xml_path}")
        return
    
    print("=" * 60)
    print("🤖 Spirobot 机器人查看器")
    print("=" * 60)
    print(f"📦 加载 Spirobot 机器人...")
    print(f"   XML 文件: {xml_path}")
    print(f"   场景: {args.scene}")
    print(f"   地面: {'✅ 已启用' if not args.no_ground else '❌ 已禁用'}")
    
    if not args.no_ground:
        print(f"   正在添加环境元素（地面、光源、背景...）")
    
    # 加载模型
    try:
        if not args.no_ground:
            # 为 XML 添加环境元素，写入临时文件以保留 mesh 相对路径
            xml_with_env = add_environment_to_xml(xml_path, add_ground=True, scene=args.scene)
            tmp_path = xml_path.parent / "_temp_spirobot_env.xml"
            tmp_path.write_text(xml_with_env, encoding="utf-8")
            model = mujoco.MjModel.from_xml_path(str(tmp_path))
        else:
            # 直接加载原始 XML
            model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # create data object now that model exists
    data = mujoco.MjData(model)

    # 若 bottle_path 存在，检查是否真的加入
    if bottle_path.exists():
        print(f"[debug] loaded model has nbody={model.nbody}")
        print(f"[debug] body names sample: {[model.body(i).name for i in range(model.nbody)][:10]}")
        # 找到名字为 "base" 的瓶子根 body 并打印位置
        try:
            for i in range(model.nbody):
                if model.body(i).name == "base":
                    pos = model.body(i).pos
                    print(f"[debug] bottle root pos: {pos}")
                    break
        except Exception:
            pass
    print(f"   关节数: {model.njnt}")
    print(f"   自由度: {model.nv}")
    print(f"   配置维度: {model.nq}")
    print(f"   控制输入: {model.nu}")
    print(f"   物体数: {model.nbody}")
    
    # 设置初始位置
    data.qpos[:] = 0  # 默认置为零

    # 如果模型中有 'base_free' 自由体，并且 body 'base' 指定了 quat
    try:
        bid = model.body('base').id
        quat = model.body(bid).quat
        # 找到绑定到 base 的 free joint
        for j in range(model.njnt):
            if model.joint(j).name == 'base_free':
                adr = int(model.joint(j).qposadr.item())
                # x,y,z同步body.pos
                data.qpos[adr:adr+3] = model.body(bid).pos
                data.qpos[adr+3:adr+7] = quat
                break
    except Exception:
        pass

    # 对于球关节，设置为单位四元数（1, 0, 0, 0）
    for i in range(model.njnt):
        joint = model.joint(i)
        if joint.type == mujoco.mjtJoint.mjJNT_BALL:
            qposadr = int(joint.qposadr.item())
            data.qpos[qposadr:qposadr+4] = [1, 0, 0, 0]  # 单位四元数
    
    mujoco.mj_forward(model, data)
    
    print("\n🎮 启动交互式查看器...")
    print("   💡 交互提示:")
    print("      • 鼠标拖拽 = 旋转视图")
    print("      • 滚轮 = 缩放")
    print("      • 右键拖拽 = 平移")
    print("      • 按 H = 查看帮助")
    print("      • 关闭窗口 = 结束演示")
    
    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机位置以看清机器人全貌
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        # 如果包含瓶子，将相机对准它
        if bottle_path.exists():
            viewer.cam.lookat = (0.2, 0, 0)
        
        # 简单的运动控制演示
        print("\n🤖 运行动作演示...")
        if args.no_ground:
            print("   (原始模型，无地面)")
        else:
            print("   (增强环境，带地面和背景)")
        
        step = 0
        max_steps = args.duration * 500  # 转换为步数（0.002s/step）
        
        while viewer.is_running() and step < max_steps:
            # 简单的周期性运动
            t = step / 100.0  # 时间参数
            
            # 第一个关节做正弦运动
            if model.nu > 0:
                data.ctrl[0] = 0.3 * np.sin(t)
            
            # 如果有更多控制输入，可以添加更多运动
            if model.nu > 1:
                data.ctrl[1] = 0.2 * np.cos(t * 0.5)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
        
        elapsed_time = step * 0.002  # 转换为秒
        print(f"\n✅ 演示结束！")
        print(f"   运行时间: {elapsed_time:.1f} 秒")
        print(f"   总步数: {step}")
    
    print("   窗口已关闭 👋")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spirobot 增强查看器")
    parser.add_argument(
        "--no_ground",
        action="store_true",
        help="不添加地面和环境（使用原始模型）"
    )
    parser.add_argument(
        "--scene",
        default="default",
        choices=["default", "bright", "dark"],
        help="场景类型"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=40,
        help="演示持续时间（秒）"
    )
    parser.add_argument(
        "--no_offset",
        action="store_true",
        help="不要对瓶子应用默认位置偏移，即完全使用 XML 中的 pos"
    )
    parser.add_argument(
        "--only",
        choices=["spirobot", "bottle", "arm"],
        default="spirobot",
        help="只加载某个模型：spirobot(默认) / bottle / arm"
    )
    
    args = parser.parse_args()
    main(args)
