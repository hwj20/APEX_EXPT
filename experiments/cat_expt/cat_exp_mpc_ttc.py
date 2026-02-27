import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import json
from datetime import datetime
import mujoco
import mujoco.viewer
import numpy as np
import cv2
from experiments.cat_expt.utils.mujoco_simulator import get_body_state, get_all_body_states
import copy

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


class TTCAgent:
    """Time-To-Collision agent: predicts collision time and evades accordingly"""
    
    def __init__(self, dt=0.01):
        self.dt = dt
        self.last_action_time = 0.0
    
    def calculate_ttc(self, robot_pos, robot_vel, cat_positions, cat_velocities, collision_threshold=0.2):
        """
        Calculate time-to-collision with nearest cat
        Returns: (ttc_time, nearest_cat_idx, collision_pos)
        """
        min_ttc = float('inf')
        nearest_cat_idx = -1
        
        for i, (cat_pos, cat_vel) in enumerate(zip(cat_positions, cat_velocities)):
            # Relative position and velocity
            rel_pos = np.array(cat_pos[:3]) - np.array(robot_pos[:3])
            rel_vel = np.array(cat_vel[:3]) - np.array(robot_vel[:3])
            
            distance = np.linalg.norm(rel_pos)
            
            # If already colliding
            if distance < collision_threshold:
                return 0.0, i, cat_pos
            
            # Calculate TTC
            if np.linalg.norm(rel_vel) > 1e-6:
                # Project relative velocity onto relative position
                approach_speed = np.dot(rel_pos, rel_vel) / distance
                if approach_speed > 0:  # Cat approaching
                    ttc = (distance - collision_threshold) / approach_speed
                    if ttc < min_ttc and ttc > 0:
                        min_ttc = ttc
                        nearest_cat_idx = i
        
        return min_ttc, nearest_cat_idx, None
    
    def decide_move(self, robot_state, cat_states, available_actions):
        """
        Decide action based on TTC
        Strategy: If TTC < threshold, perform evasion
        """
        ttc, cat_idx, _ = self.calculate_ttc(
            robot_state["position"],
            robot_state["velocity"],
            [cat["position"] for cat in cat_states],
            [cat["velocity"] for cat in cat_states]
        )
        
        # Evasion strategy based on threat level
        if ttc < 1.0:  # Critical: collision imminent
            # Move away from nearest cat
            if cat_idx >= 0:
                cat_pos = np.array(cat_states[cat_idx]["position"][:2])
                robot_pos = np.array(robot_state["position"][:2])
                diff = robot_pos - cat_pos
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    unit_diff = diff / norm
                    # Choose action that maximizes distance
                    if abs(unit_diff[0]) > abs(unit_diff[1]):
                        move = "move_left" if unit_diff[0] < 0 else "move_right"
                    else:
                        move = "move_down" if unit_diff[1] < 0 else "move_up"
                else:
                    move = "jump"  # Jump as emergency evasion
            else:
                move = "jump"
        elif ttc < 2.0:  # Warning: collision likely
            move = "jump"
        else:  # Safe
            move = "stay"
        
        return available_actions[move], True


class MPCAgent:
    """Model Predictive Control agent: simulates future actions and chooses optimal"""
    
    def __init__(self, dt=0.01, prediction_horizon=10):
        self.dt = dt
        self.prediction_horizon = prediction_horizon
        self.collision_threshold = 0.2
    
    def simulate_action(self, model, initial_data, action_name, action_spec, num_steps):
        """Simulate action and return final state and collision info"""
        sim_data = copy.deepcopy(initial_data)
        
        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        dof_start = model.body_dofadr[robot_body_id]
        
        vel = action_spec["velocity"]
        for i in range(3):
            sim_data.qvel[dof_start + i] = vel[i]
        
        collision_detected = False
        closest_distance = float('inf')
        
        for _ in range(num_steps):
            mujoco.mj_step(model, sim_data)
            
            # Check collision with any cat
            robot_pos = sim_data.xpos[robot_body_id].copy()
            for i in range(1, 5):  # up to 4 cats
                try:
                    cat_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cat{}".format(i))
                    if cat_body_id >= 0:
                        cat_pos = sim_data.xpos[cat_body_id].copy()
                        distance = np.linalg.norm(robot_pos[:3] - cat_pos[:3])
                        closest_distance = min(closest_distance, distance)
                        if distance < self.collision_threshold:
                            collision_detected = True
                except:
                    pass
        
        final_state = get_all_body_states(model, sim_data)
        return {
            "action": action_name,
            "collision": collision_detected,
            "closest_distance": closest_distance,
            "final_state": final_state,
            "data": sim_data
        }
    
    def evaluate_action(self, model, initial_data, action_name, action_spec):
        """Evaluate action safety score"""
        num_steps = int(action_spec["duration"] / self.dt)
        sim_result = self.simulate_action(model, initial_data, action_name, action_spec, num_steps)
        
        # Score: higher is better (prioritizes distance to collision)
        if sim_result["collision"]:
            score = -1000
        else:
            score = sim_result["closest_distance"]
        
        return score, sim_result
    
    def decide_move(self, robot_state, cat_states, available_actions, model, data):
        """
        Evaluate all available actions and select the one with best collision score
        """
        best_score = -float('inf')
        best_action = None
        
        for action_name, action_spec in available_actions.items():
            score, _ = self.evaluate_action(model, data, action_name, action_spec)
            if score > best_score:
                best_score = score
                best_action = available_actions[action_name]
        
        action_valid = best_score >= -500  # Consider valid if not immediate collision
        return best_action, action_valid


def run_exp(difficulty, method='MPC', prediction_horizon=10, run_callback=None):
    """Run experiment with MPC or TTC method"""
    
    physical_model = None
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
            data.qvel[0:3] = vel

    cat_num, cat_speed = 2, 1.0
    if difficulty == 'Simple':
        cat_speed = 1.0
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
    file_name = f"turing_cat_{difficulty}_{method}_horizon{prediction_horizon}_{timestamp}.mp4"
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"),
                                   fps,
                                   (width, height))

    data = mujoco.MjData(physical_model)
    renderer = None
    rendering_available = False
    
    # Try to initialize renderer for video rendering
    try:
        renderer = mujoco.Renderer(physical_model, height=height, width=width)
        rendering_available = True
    except Exception as e:
        print(f"Warning: Rendering not available ({e}). Video will contain blank frames.")
        print("To enable rendering, use: xvfb-run -a -s '-screen 0 1920x1080x24' python script.py")

    # Initialize agent
    if method == 'MPC':
        agent = MPCAgent(dt=1.0/fps, prediction_horizon=prediction_horizon)
    elif method == 'TTC':
        agent = TTCAgent(dt=1.0/fps)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Begin Simulation
    current_action = None
    frames_left = 0
    action_index = -1
    action_sequence = []
    dt = 1.0 / fps
    init_frames = int(fps / 2)
    response_time = 0.0

    snapshot_t_dt = None

    # Simulation: 10s
    for step in range(10 * fps):
        # Current State
        robot_state = get_body_state(physical_model, data, "robot")
        cat_states = []
        for i in range(1, cat_num + 1):
            cat_states.append(get_body_state(physical_model, data, f"cat{i}"))

        new_action = ""
        action_valid = True

        # Turn cats towards the Agent every second
        if step % fps == 0:
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
                collision = True

        if step > init_frames and frames_left <= 0:
            start = time.perf_counter()
            if method == 'MPC':
                move, action_valid = agent.decide_move(robot_state, cat_states, available_move, 
                                                       physical_model, data)
            elif method == 'TTC':
                move, action_valid = agent.decide_move(robot_state, cat_states, available_move)
            end = time.perf_counter()
            response_time = end - start
            
            try:
                add_action(move)
                new_action = move
            except Exception as e:
                print(f"Error setting move: {e}")

        # Execute current move
        if frames_left > 0:
            frames_left -= 1
            if frames_left <= 0 and action_index + 1 < len(action_sequence):
                action_index += 1
                current_action = action_sequence[action_index]
                frames_left = int(current_action["duration"] * fps)
                vel = current_action["velocity"]
                data.qvel[0:3] = vel
        else:
            data.qvel[0:3] = [0.0, 0.0, 0.0]

        if run_callback is not None:
            run_callback(step / fps, collision, new_action, action_valid, response_time)

        # Step physics
        mujoco.mj_step(physical_model, data)

        # render
        if rendering_available and renderer:
            try:
                renderer.update_scene(data, camera="top")
                pixels = renderer.render()
                frame = cv2.cvtColor(np.flipud(pixels), cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame, (width, height))
            except Exception as e:
                # Fallback: create a blank frame if rendering fails
                frame_resized = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Create a blank frame
            frame_resized = np.zeros((height, width, 3), dtype=np.uint8)
        
        video_writer.write(frame_resized)

    video_writer.release()
    print(f"Video saved as {file_name}")
    return collision


def run_trial(difficulty, method, prediction_horizon, trial_id):
    print(f"\nRunning: {difficulty} | {method} (horizon={prediction_horizon}) | Trial {trial_id + 1}")
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

    collision_flag = run_exp(difficulty, method=method, prediction_horizon=prediction_horizon, 
                            run_callback=callback_fn)

    return {
        "survival_time": survival_time,
        "collision_flag": collision_flag,
        "invalid_actions": invalid_actions,
        "valid_actions": valid_actions,
        "latency_sum": response_time_sum,
        "actions_len": valid_actions + invalid_actions
    }


if __name__ == "__main__":
    EXPT_TIME = 10  # seconds
    NUM_TRIALS = 5
    difficulties = ["Simple", "Medium", "Hard"]
    methods = {
        "MPC": [5, 10],  # different prediction horizons
        "TTC": [1]  # TTC doesn't use horizon parameter
    }
    
    results = {
        d: {
            m: {
                str(h): {
                    "cfr": 0,
                    "ast": [],
                    "iar": 0,
                    "latency": []
                } for h in methods[m]
            } for m in methods
        } for d in difficulties
    }
    save_path = "results/results_mpc_ttc.json"

    for difficulty in difficulties:
        for method in methods:
            for horizon in methods[method]:
                for i in range(NUM_TRIALS):
                    result = run_trial(difficulty, method, horizon, i)
                    results[difficulty][method][str(horizon)]["ast"].append(result["survival_time"])
                    results[difficulty][method][str(horizon)]["cfr"] += int(not result["collision_flag"])

                    if result["actions_len"] > 0:
                        results[difficulty][method][str(horizon)]["iar"] += result["invalid_actions"] / result["actions_len"] / NUM_TRIALS
                        results[difficulty][method][str(horizon)]["latency"].append(
                            result["latency_sum"] / result["actions_len"])
                    else:
                        results[difficulty][method][str(horizon)]["iar"] += 0
                        results[difficulty][method][str(horizon)]["latency"].append(0.0)

                with open(save_path, "w") as f:
                    json.dump(results, f, indent=4)

    print("\n" + "="*80)
    print("Final Results Summary:")
    print("="*80)
    for difficulty in difficulties:
        print(f"\n[{difficulty}]")
        for method in methods:
            for horizon in methods[method]:
                summary = results[difficulty][method][str(horizon)]
                avg_ast = np.mean(summary["ast"]) if summary["ast"] else 0
                avg_latency = np.mean(summary["latency"]) if summary["latency"] else 0
                cfr_rate = summary["cfr"] / NUM_TRIALS
                print(f"  {method} (H={horizon}): CFR={cfr_rate:.2f} | AST={avg_ast:.2f}s | "
                      f"IAR={summary['iar']:.2f} | Avg Latency={avg_latency:.4f}s")
