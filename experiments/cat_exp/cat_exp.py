import json

import mujoco
import mujoco.viewer
import numpy as np
import cv2
from experiments.cat_exp.utils.cat_game_agent import LLM_Agent
from experiments.cat_exp.utils.APEX import APEX
from experiments.cat_exp.model.graphormer import DiffGraphormer
from experiments.cat_exp.utils.mujoco_perception import get_body_state, get_all_body_states
import torch

with open('env/square_demo.xml', 'r') as f:
    square_xml = f.read()
with open('env/simple_env.xml', 'r') as f:
    simple_xml = f.read()
with open('env/medium_env.xml', 'r') as f:
    medium_xml = f.read()
with open('env/hard_env.xml', 'r') as f:
    hard_xml = f.read()
with open("env/available_move.json", 'r') as f:
    available_move = json.load(f)


def move_cat_towards_robot(cat_pos, robot_pos, speed=0.03):
    direction = np.array(robot_pos) - np.array(cat_pos)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0.0, 0.0
    unit_dir = direction / norm
    vx, vy = unit_dir * speed
    return vx, vy


def run_exp(difficulty, method='APEX', model='gpt-4o-mini'):
    physical_model = None
    agent = LLM_Agent(model=model)

    def add_action(_move):
        nonlocal current_action, frames_left, action_index
        action_sequence.append(_move)
        if frames_left <= 0 and action_index + 1 < len(action_sequence):
            action_index += 1
            current_action = action_sequence[action_index]
            frames_left = int(current_action["duration"] * fps)

    cat_num, cat_speed = 2, 1.0
    if difficulty == 'Simple':
        cat_speed = 1.0
        # physical_model = mujoco.MjModel.from_xml_string(simple_xml)
        physical_model = mujoco.MjModel.from_xml_string(square_xml)
        cat_num = 2
    elif difficulty == "Medium":
        cat_speed = 2.0
        physical_model = mujoco.MjModel.from_xml_string(medium_xml)
        cat_num = 3
    elif difficulty == "Hard":
        cat_speed = 3.0
        physical_model = mujoco.MjModel.from_xml_string(hard_xml)
        cat_num = 4

    # video setting
    fps = 100
    width, height = 640, 480
    video_writer = cv2.VideoWriter(f"turing_cat_llm_{difficulty}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                   (width, height))

    # Initialize cats and env
    cats = [
        {"id": 1, "vx": 3, "vy": 2},
        {"id": 2, "vx": -2, "vy": 4}
    ]
    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=height, width=width)
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
    init_frames = int(fps / 2)

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
        danger_model.load_state_dict(torch.load('model/diffgraphormer_physics.pt', map_location='cpu'))
        danger_model.eval()  # Turn off dropout
        apex = APEX(graphormer_model=danger_model, physics_simulator="mujoco", llm_agent=agent, dt=dt,
                    available_move=available_move)

    snapshot_t, snapshot_t_dt = None, None

    # Simulation: 10s
    for step in range(10 * fps):
        # Current State
        robot_state = get_body_state(physical_model, data, "robot")
        cat1_state = get_body_state(physical_model, data, "cat1")
        cat2_state = get_body_state(physical_model, data, "cat2")

        if step % fps == 0:
            # Turn cats towards the Agent
            for i in range(1, cat_num + 1):
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
            np.array(robot_state["position"][:3]) - np.array(cat1_state["position"][:3]))
        cat_distance_2 = np.linalg.norm(
            np.array(robot_state["position"][:3]) - np.array(cat2_state["position"][:3]))

        if step > init_frames and (cat_distance_1 <= 0.2 or cat_distance_2 <= 0.2):
            print(step)
            print(cat1_state)
            print(cat2_state)
            print(robot_state)
            print("Collision!")

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}
        if step > init_frames and frames_left <= 0:
            if method == 'APEX':
                if snapshot_t and snapshot_t != snapshot_t_dt:
                    triggered, move = apex.run(snapshot_t, snapshot_t_dt, dt, physical_model, data, step)
                    if triggered:
                        add_action(move)

            if method == 'LLM' and step % fps == 0:
                response = agent.decide_move(get_all_body_states(physical_model, data), available_move)
                try:
                    move = response
                    add_action(move)
                except:
                    print("error setting move")
            if method == 'TEST' and step % fps == 0:
                move = {'velocity': [0.0, 0.0, 3.0], 'duration': 0.1}
                add_action(move)
        snapshot_t = snapshot_t_dt
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

        # print(robot_state)

        # Step
        mujoco.mj_step(physical_model, data)

        # render
        renderer.update_scene(data, camera="top")
        pixels = renderer.render()
        frame = cv2.cvtColor(np.flipud(pixels), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (width, height))
        video_writer.write(frame_resized)

    video_writer.release()
    print(f"turing_cat_llm_{difficulty}.mp4")


run_exp("Simple", method="TEST")
# run_exp("Medium")
# run_exp("Hard")
