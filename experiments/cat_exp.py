import json

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from experiments.utils.cat_game_agent import LLM_Agent
from experiments.utils.APEX import APEX
from model.graphormer import DiffGraphormer
import torch

simple_xml = """
<mujoco> 
    <option gravity="0 0 1" integrator="RK4" timestep="0.01" />
    
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


def get_all_body_states(model, data, filter_out=['world']):
    states = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name not in filter_out:
            pos = data.xpos[i].copy()
            vel = data.cvel[i, :6].copy()
            states.append({
                "name": name,
                "position": pos.tolist(),
                "velocity": vel.tolist()
            })
    return states


def run_exp(difficulty, method='APEX', model='gpt-4o-mini'):
    physical_model = None
    agent = LLM_Agent(model=model)

    def add_action(move):
        nonlocal current_action, frames_left, action_index
        action_sequence.append(move)
        if frames_left <= 0 and action_index + 1 < len(action_sequence):
            action_index += 1
            current_action = action_sequence[action_index]
            frames_left = int(current_action["duration"] * fps)

    if difficulty == 'Simple':
        cat_speed = 1.0
        physical_model = mujoco.MjModel.from_xml_string(simple_xml)
    elif difficulty == "Medium":
        cat_speed = 2.0
        physical_model = mujoco.MjModel.from_xml_string(Medium_Hard_xml)
    elif difficulty == "Hard":
        cat_speed = 3.0
        physical_model = mujoco.MjModel.from_xml_string(Medium_Hard_xml)

    # video setting
    fps = 30
    width, height = 640, 480
    video_writer = cv2.VideoWriter(f"turing_cat_llm_{difficulty}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                   (width, height))

    # Initialize cats and env
    cats = [
        {"id": 1, "vx": 3, "vy": 2},
        {"id": 2, "vx": -2, "vy": 4}
    ]
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model)
    for cat in cats:
        cid = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, "cat" + str(cat['id']))
        cid -= 1
        data.qvel[cid * 6] += cat["vx"]
        data.qvel[cid * 6 + 1] += cat["vy"]

    # Begin Simulation
    current_action = None
    frames_left = 0
    action_index = -1
    action_sequence = []
    dt = 1.0 / fps  # unit: seconds

    # Initialize Model
    apex = None
    if method == 'APEX':
        danger_model = DiffGraphormer(
            in_feats=7,  # 输入维度要一致
            edge_feat_dim=3,
            hidden_dim=32,  # 你训练时的 hidden size
            num_heads=4,
            dropout=0.3
        )
        danger_model.load_state_dict(torch.load('./model/diffgraphormer_physics.pt', map_location='cpu'))
        danger_model.eval()  # 推理模式（关闭dropout）
        apex = APEX(graphormer_model=danger_model, physics_simulator="mujoco", llm_agent=agent, dt=dt)

    snapshot_t, snapshot_dt = None, None

    # Simulation: 10s
    for step in range(300):
        # Current State
        robot_state = get_body_state(physical_model, data, "robot")
        cat1_state = get_body_state(physical_model, data, "cat1")
        cat2_state = get_body_state(physical_model, data, "cat2")

        # TODO 1 second 1 time
        # Turn cats towards the Agent
        for i in [1, 2]:  # cat1 和 cat2
            cat_name = f"cat{i}"
            cat_state = get_body_state(physical_model, data, cat_name)
            vx, vy = move_cat_towards_robot(cat_state["position"][:2], robot_state["position"][:2],
                                            speed=cat_speed)
            cat_body_id = mujoco.mj_name2id(physical_model, mujoco.mjtObj.mjOBJ_BODY, cat_name)
            cat_dof_start = physical_model.body_dofadr[cat_body_id]
            data.qvel[cat_dof_start] = vx
            data.qvel[cat_dof_start + 1] = vy

        # Collision Check
        cat_distance_1 = np.linalg.norm(
            np.array(robot_state["position"][:2]) - np.array(cat1_state["position"][:2]))
        cat_distance_2 = np.linalg.norm(
            np.array(robot_state["position"][:2]) - np.array(cat2_state["position"][:2]))

        if cat_distance_1 <= 0.2 or cat_distance_2 <= 0.2:
            print(cat1_state)
            print(cat2_state)
            print("🚨 Danger! 猫猫撞上robot啦！")

        if step > 10 and frames_left <= 0:
            if method == 'APEX':
                snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}
                if snapshot_t and snapshot_t != snapshot_t_dt:
                    triggered, move = apex.run(snapshot_t, snapshot_t_dt, dt, physical_model, data, step)
                    if triggered:
                        add_action(move)
                snapshot_t = snapshot_t_dt

            if method == 'LLM':
                response = agent.decide_move(get_all_body_states(physical_model, data))
                try:
                    move = json.loads(response)
                    add_action(move)
                except:
                    print("error setting move")

        # Execute current move
        if frames_left > 0:
            vel = current_action["velocity"]
            data.qvel[0:3] = vel

            frames_left -= 1

            # Current Action ends and moves to the next action
            if frames_left <= 0 and action_index + 1 < len(action_sequence):
                action_index += 1
                current_action = action_sequence[action_index]
                frames_left = int(current_action["duration"] * fps)
        else:
            data.qvel[0:3] = [0.0, 0.0, 0.0]

        print(robot_state)
        # Step
        mujoco.mj_step(physical_model, data)

        # render
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
