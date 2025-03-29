import json
import re

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from experiments.cat_game_agent import LLM_Agent


def strip_markdown(text: str) -> str:
    # 去除代码块标记 ```json ```python等
    text = re.sub(r"```", "", text)
    text = re.sub("json", "", text)
    # 去除标题 #
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # 去除加粗/斜体 ** ** 或 __ __ 或 * *
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    # 去除行内代码 `
    text = re.sub(r"`(.*?)`", r"\1", text)
    # 去除列表项 - *
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    # 去除多余空行
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

simple_xml = """
<mujoco> 
    <option gravity="0 0 0" integrator="RK4" timestep="0.01" />
    
    <visual>
        <map znear="0.01"/>
    </visual>

    <asset>
        <!-- 添加网格纹理 -->
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.8 0.8 0.8" rgb2="1 1 1"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.2"/>
    </asset>

    <worldbody>
    <camera name="top" pos="0 0 5" xyaxes="1 0 0 0 1 0"/>
        <!-- 添加地板（平面 + 网格材质） -->
        <geom name="floor" type="plane" size="5 5 0.1" material="grid_mat" rgba="1 1 1 1"/>

        <body name="robot" pos="0 0 0.1">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
        </body>
        <body name="cat1" pos="1 1 0.1">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
        <body name="cat2" pos="-1 -1 0.1">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

Simple_xml = """
<mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="robot" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
        </body>
        <body name="cat1" pos="1 1 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
        <body name="cat2" pos="-1 -1 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""

Medium_Hard_xml = """
<mujoco>
<option gravity="0 0 0"/>
<worldbody>
    <!-- Robot -->
    <body name="robot" pos="0 0 0">
        <freejoint/>
        <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
    </body>
    <!-- Cats -->
    <body name="cat1" pos="1 1 0">
        <freejoint/>
        <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
    <body name="cat2" pos="-1 -1 0">
        <freejoint/>
        <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>

    <!-- Obstacles -->
    <geom name="obstacle1" type="box" pos="0.3 0 0.05" size="0.05 0.3 0.05" rgba="0.6 0.6 0.6 1"/>
    <geom name="obstacle2" type="box" pos="-0.3 0 0.05" size="0.05 0.3 0.05" rgba="0.6 0.6 0.6 1"/>
</worldbody>
</mujoco>
"""


def move_cat_towards_robot(cat_pos, robot_pos, speed=0.03):
    direction = np.array(robot_pos) - np.array(cat_pos)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0.0, 0.0
    unit_dir = direction / norm
    vx, vy = unit_dir * speed
    return vx, vy


def get_body_state(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = data.xpos[body_id].copy()
    vel = data.cvel[body_id, :6].copy()
    return {
        "name": body_name,
        "position": pos.tolist(),
        "velocity": vel.tolist()
    }


def get_all_body_states(model, data):
    states = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:  # 有名字才记录
            pos = data.xpos[i].copy()
            vel = data.cvel[i, :6].copy()
            states.append({
                "name": name,
                "position": pos.tolist(),
                "velocity": vel.tolist()
            })
    return states


def run_exp(difficulty):
    if difficulty == 'Simple':
        cat_speed = 3.0
        # ✅ 创建 MuJoCo 物理环境
        model = mujoco.MjModel.from_xml_string(simple_xml)
    elif difficulty == "Medium":
        cat_speed = 3.0
        # ✅ 创建 MuJoCo 物理环境
        model = mujoco.MjModel.from_xml_string(Medium_Hard_xml)
    elif difficulty == "Hard":
        cat_speed = 5.0
        # ✅ 创建 MuJoCo 物理环境
        model = mujoco.MjModel.from_xml_string(Medium_Hard_xml)

    # ✅ 视频设置
    fps = 30
    width, height = 640, 480
    video_writer = cv2.VideoWriter(f"turing_cat_llm_{difficulty}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                   (width, height))

    agent = LLM_Agent(model='gpt-4o-mini')
    # ✅ 初始速度设置
    cats = [
        {"id": 1, "vx": 3, "vy": 2},
        {"id": 2, "vx": -2, "vy": 4}
    ]
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    # ✅ 模拟循环
    current_action = None
    frames_left = 0
    action_index = 0
    action_sequence = []
    dt = 1.0 / fps  # 每帧时长（单位：秒）

    for cat in cats:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cat" + str(cat['id']))
        cid -= 1
        data.qvel[cid * 6] += cat["vx"]
        data.qvel[cid * 6 + 1] += cat["vy"]

    # ✅ 模拟循环 10 s
    for step in range(300):

        # 提取当前状态
        robot_state = get_body_state(model, data, "robot")
        cat1_state = get_body_state(model, data, "cat1")
        cat2_state = get_body_state(model, data, "cat2")
        print(cat1_state)
        print(cat2_state)
        # ✅ 每帧调整猫猫速度：让它们朝机器人冲
        for i in [1, 2]:  # cat1 和 cat2
            cat_name = f"cat{i}"
            cat_state = get_body_state(model, data, cat_name)
            vx, vy = move_cat_towards_robot(cat_state["position"][:2], robot_state["position"][:2],
                                            speed=cat_speed)  # 你可以调一下这个速度
            cat_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
            cat_dof_start = model.body_dofadr[cat_body_id]  # 每个 body 的自由度起始索引
            data.qvel[cat_dof_start] = vx
            data.qvel[cat_dof_start + 1] = vy

        # 每隔一段时间请求新的动作序列（或首次）
        # if frames_left <= 0:
        #     action_index = 0
        #     response = agent.decide_move(get_all_body_states(model, data))
        #     response = strip_markdown(response)
        #     try:
        #         action_sequence = json.loads(response)
        #         print(action_sequence)
        #     except:
        #         action_sequence = []
        #
        #     if action_sequence:
        #         current_action = action_sequence[action_index]
        #         frames_left = int(current_action["duration"] * fps)  # convert duration to frame count

        # 执行当前动作（持续一段时间）
        if current_action and frames_left > 0:
            ax, ay = current_action["acceleration"]
            data.qvel[0] += ax * dt  # 加速度 × dt = Δv
            data.qvel[1] += ay * dt

            frames_left -= 1

            # 当前动作结束，切到下一个
            if frames_left <= 0 and action_index + 1 < len(action_sequence):
                action_index += 1
                current_action = action_sequence[action_index]
                frames_left = int(current_action["duration"] * fps)

        print(robot_state)
        # 模拟一步
        mujoco.mj_step(model, data)

        # 渲染并保存帧
        renderer.update_scene(data)
        pixels = renderer.render()
        frame = cv2.cvtColor(np.flipud(pixels), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (width, height))
        video_writer.write(frame_resized)

    video_writer.release()
    print(f"turing_cat_llm_{difficulty}.mp4")


run_exp("Simple")
# run_exp("Medium")
# run_exp("hard")
