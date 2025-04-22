import copy
import mujoco
import numpy as np


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


class simulator:
    def __init__(self, method):
        self.method = method

    def mujoco_sim(self, model, env_data, available_moves):
        sim_results = {}
        max_duration = 1.0
        timestep = model.opt.timestep
        max_steps = int(max_duration / timestep)

        for action_name, action_desc in available_moves.items():
            sim_data = copy.deepcopy(env_data)

            robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            dof_start = model.body_dofadr[robot_body_id]

            vel = [0.0, 0.0, 0.0]  # Can be replaced with the real vel of the robot
            if action_name == "move_left":
                vel[0] = -3.0
            elif action_name == "move_right":
                vel[0] = 3.0
            elif action_name == "move_up":
                vel[1] = -3.0
            elif action_name == "move_down":
                vel[1] = 3.0
            elif action_name == "jump":
                vel[2] = 3.0

            for i in range(3):
                sim_data.qvel[dof_start + i] = vel[i]

            safe_steps = 0
            last_pos = np.array(sim_data.xpos[robot_body_id])

            for step in range(max_steps):
                mujoco.mj_step(model, sim_data)
                current_pos = np.array(sim_data.xpos[robot_body_id])
                movement = np.linalg.norm(current_pos[:3] - last_pos[:3])
                if movement < 1e-3 and action_name not in ['jump', 'stay']:
                    break

                last_pos = current_pos
                safe_steps += 1

            safe_duration = safe_steps * timestep
            # stay away from wall
            if safe_duration < 1.0 and action_name not in ['jump', 'stay']:
                safe_duration -= 0.1
                # if save_duration < 0, then the action will be evaluated into 'invalid' in the summary
            else:
                safe_duration = 1.0

            sim_results[action_name] = {
                "final_pos": get_all_body_states(model, sim_data),
                "description": {
                    "velocity": vel,
                    "duration": safe_duration,
                    "description": f"{action_name} with velocity={vel} for {safe_duration:.2f}s"
                }
            }

        return sim_results

    def sim(self, model, env_data, action):
        if self.method == 'mujoco':
            return self.mujoco_sim(model, env_data, action)
        return None
