import time
import json
from datetime import datetime
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from experiments.cat_expt.utils.cat_game_agent_other_llms import LLM_Agent
from experiments.cat_expt.utils.APEX import APEX
from experiments.cat_expt.model.graphormer import DiffGraphormer
from experiments.cat_expt.utils.mujoco_simulator import get_body_state, get_all_body_states
import torch

with open('experiments/cat_expt/env/square_demo.xml', 'r') as f:
    square_xml = f.read()
with open('experiments/cat_expt/env/simple_env.xml', 'r') as f:
    simple_xml = f.read()
with open('experiments/cat_expt/env/medium_env.xml', 'r') as f:
    medium_xml = f.read()
with open('experiments/cat_expt/env/hard_env.xml', 'r') as f:
    hard_xml = f.read()
with open("experiments/cat_expt/env/available_move.json", 'r') as f:
    available_move = json.load(f)


def move_cat_towards_robot(cat_pos, robot_pos, speed=0.03):
    direction = np.array(robot_pos) - np.array(cat_pos)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0.0, 0.0
    unit_dir = direction / norm
    vx, vy = unit_dir * speed
    return vx, vy


def run_exp(difficulty, method='APEX', model='gpt-4o-mini', run_callback=None):
    # I know how ugly the code is :)
    # if method == 'VLM':
    #     model = 'gpt-4o'

    model_name = model.replace('/','_')
    physical_model = None
    agent = LLM_Agent(model=model)
    collision = False
    collision_threshold = 0.2

    def add_action(_move):
        nonlocal current_action, frames_left, action_index
        action_sequence.append(_move)
        if frames_left <= 0 and action_index + 1 < len(action_sequence):
            action_index += 1
            current_action = action_sequence[action_index]
            frames_left = int(current_action["duration"] * fps)
            vel = current_action["velocity"]
            data.qvel[0:3] = vel  # robot_index:0

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"turing_cat_llm_{difficulty}_{method}_{model_name}_{timestamp}.mp4"
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps,
                                   (width, height))

    # Initialize cats and env

    data = mujoco.MjData(physical_model)
    renderer = mujoco.Renderer(physical_model, height=height, width=width)

    # Begin Simulation
    current_action = None
    frames_left = 0
    action_index = -1
    action_sequence = []
    dt = 1.0 / fps  # unit: seconds
    init_frames = int(fps / 2)
    response_time = 0.0

    # Initialize Model
    apex = None
    if method == 'APEX':
        danger_model = DiffGraphormer(
            in_feats=7,
            edge_feat_dim=3,
            hidden_dim=32,
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
        cat_states = []
        for i in range(1, cat_num + 1):
            cat_states.append(get_body_state(physical_model, data, f"cat{i}"))

        new_action = ""
        action_valid = True

        if step % fps == 0:
            # Turn cats towards the Agent every second
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
        for cat_state in cat_states:
            cat_distance = np.linalg.norm(
                np.array(robot_state["position"][:3]) - np.array(cat_state["position"][:3]))
            if step > init_frames and cat_distance < collision_threshold:
                # print(step)
                # print("Collision!")
                collision = True

        snapshot_t_dt = {"objects": get_all_body_states(physical_model, data)}
        if step > init_frames and frames_left <= 0:
            if method == 'APEX':
                if snapshot_t and snapshot_t != snapshot_t_dt:
                    start = time.perf_counter()
                    triggered, move, action_valid = apex.run(snapshot_t, snapshot_t_dt, dt, physical_model, data, step)
                    end = time.perf_counter()
                    response_time = end - start
                    if triggered:
                        add_action(move)
                        new_action = move

            if method == 'LLM' and step % fps == 0:
                start = time.perf_counter()
                move, action_valid = agent.decide_move(get_all_body_states(physical_model, data), available_move)
                end = time.perf_counter()
                response_time = end - start
                try:
                    add_action(move)
                    new_action = move
                except:
                    print(move)
                    print("error setting move")

            if method == 'VLM' and step % fps == 0:
                start = time.perf_counter()
                image_path = f"experiments/cat_expt/tmp/frame_{difficulty}_{method}_{model_name}_step{step}.png"
                renderer.update_scene(data, camera="top")
                pixels = renderer.render()
                frame_rgb = np.flipud(pixels)
                cv2.imwrite(image_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                move, action_valid = agent.decide_move_vlm(get_all_body_states(physical_model, data), available_move,
                                                           image_path)
                end = time.perf_counter()
                response_time = end - start
                try:
                    add_action(move)
                    new_action = move
                except:
                    print(move)
                    print("error setting move")

            if method == 'TEST' and step % fps == 0:
                move = {'velocity': [0.0, 0.0, 3.0], 'duration': 0.1}
                add_action(move)
        snapshot_t = snapshot_t_dt
        # Execute current move
        if frames_left > 0:
            frames_left -= 1

            # Current Action ends and moves to the next action
            if frames_left <= 0 and action_index + 1 < len(action_sequence):
                action_index += 1
                current_action = action_sequence[action_index]
                frames_left = int(current_action["duration"] * fps)
                vel = current_action["velocity"]
                data.qvel[0:3] = vel  # robot_index:0
        else:
            data.qvel[0:3] = [0.0, 0.0, 0.0]

        # print(robot_state)
        if run_callback is not None:
            run_callback(step / fps, collision, new_action, action_valid, response_time)

        # Step
        mujoco.mj_step(physical_model, data)

        # render
        renderer.update_scene(data, camera="top")
        pixels = renderer.render()
        frame = cv2.cvtColor(np.flipud(pixels), cv2.COLOR_RGB2BGR)
        frame_resized = cv2.resize(frame, (width, height))
        video_writer.write(frame_resized)

    video_writer.release()
    print(f"video saved as {file_name}")
    return collision


def run_trial(difficulty, method, model, trial_id):
    print(f"\nRunning: {difficulty} | {method} | {model} | Trial {trial_id + 1}")
    survival_time = EXPT_TIME
    collision_times = 0
    invalid_actions = 0
    valid_actions = 0
    actions = {}
    response_time_sum = 0.0

    def callback_fn(current_time, collided, new_action, action_valid, response_time):
        nonlocal survival_time, collision_times, invalid_actions, valid_actions, response_time_sum
        if survival_time == EXPT_TIME and collided:
            survival_time = current_time
        if collided:
            collision_times += 1
        if not action_valid:
            invalid_actions += 1
        if new_action != "":
            if action_valid:
                valid_actions += 1
            actions[survival_time] = (new_action, response_time)
            response_time_sum += response_time

    collision_flag = run_exp(difficulty, method=method, model=model, run_callback=callback_fn)

    return {
        "survival_time": survival_time,
        "collision_flag": collision_flag,
        "invalid_actions": invalid_actions,
        "valid_actions": valid_actions,
        "latency_sum": response_time_sum,
        "actions_len": valid_actions+invalid_actions
    }


if __name__ == "__main__":
    # run_exp(difficulty='Simple', method='LLM', model='gemini2.5-flash')
    # run_exp(difficulty='Medium', method='VLM', model='gpt-4o-mini')
    # run_exp(difficulty='Hard', method='VLM', model='gpt-4o-mini')

    EXPT_TIME = 10  # seconds
    NUM_TRIALS = 5
    difficulties = [
                    "Simple",
                     "Medium", 
                     "Hard"
                     ]
    methods = {
    "VLM": [
    "claude-sonnet-4-20250514",  # Claude 4
    "gemini-2.5-flash",              # Gemini 2.5（Google）
    "meta-llama/llama-4-scout",  # HuggingFace LLaMA 4
     "gpt-4.1",      # OpenAI 4.1
    ],
        "LLM": [
    "deepseek/deepseek-r1-0528",           # DeepSeek r1
    "claude-sonnet-4-20250514",  # Claude 4
    "gemini-2.5-flash",              # Gemini 2.5（Google）
    "meta-llama/llama-4-scout",  # HuggingFace LLaMA 4
    "gpt-4.1",      # OpenAI 4.1
]}

    results = {
        d: {
            m: {
                model: {
                    "cfr": 0,
                    "ast": [],
                    "iar": 0,
                    "latency": []
                } for model in methods[m]
            } for m in methods
        } for d in difficulties
    }
    save_path = "experiments/cat_expt/results/results_other_llm.json"

    for difficulty in difficulties:
        for method in methods:
            for model in methods[method]:
                for i in range(NUM_TRIALS):
                    result = run_trial(difficulty, method, model, i)
                    results[difficulty][method][model]["ast"].append(result["survival_time"])
                    results[difficulty][method][model]["cfr"] += int(not result["collision_flag"])

                    if result["actions_len"] > 0:
                        results[difficulty][method][model]["iar"] += result["invalid_actions"] / result["actions_len"]/NUM_TRIALS
                        results[difficulty][method][model]["latency"].append(
                            result["latency_sum"] / result["actions_len"])
                    else:
                        results[difficulty][method][model]["iar"] += 0
                        results[difficulty][method][model]["latency"].append(0.0)

                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)

    print("\nFinal Results Summary:\n")
    for difficulty in difficulties:
        for method in methods:
            for model in methods[method]:
                summary = results[difficulty][method][model]
                avg_ast = np.mean(summary["ast"]) if summary["ast"] else 0
                avg_latency = np.mean(summary["latency"]) if summary["latency"] else 0
                cfr_rate = summary["cfr"] / NUM_TRIALS
                print(f"[{difficulty}][{method}][{model}] CFR={cfr_rate:.2f} | AST={avg_ast:.2f}s | "
                      f"IAR={summary['iar']:.2f} | Avg Latency={avg_latency:.2f}s")
