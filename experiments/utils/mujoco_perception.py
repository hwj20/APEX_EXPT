import copy
import json

import mujoco
import numpy as np


# Helper: create mujoco model with a single sphere
def create_basic_model(r):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
            <body name="ball" pos="{r} 0 0">
                <freejoint/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """)


# Simulate 3D linear motion with manual integration
def simulate_3d_linear_motion(v0, a, t):
    model = create_basic_model(0)
    data = mujoco.MjData(model)

    dt = 0.001
    steps = int(t / dt)
    for i in range(steps):
        for j in range(3):
            data.qvel[j] = a[j] * dt * i + v0[j]
            data.qpos[j] = v0[j] * dt * i + 0.5 * a[j] * (dt * i) ** 2
        mujoco.mj_step(model, data)

    return {
        "velocity_x": round(data.qvel[0], 4),
        "velocity_y": round(data.qvel[1], 4),
        "velocity_z": round(data.qvel[2], 4),
        "displacement_x": round(data.qpos[0], 4),
        "displacement_y": round(data.qpos[1], 4),
        "displacement_z": round(data.qpos[2], 4)
    }


# Simulate 3D circular motion manually
def simulate_3d_circular_motion(p):
    r = p["r"]
    omega = p["omega"]
    t_total = p["t"]
    plane = p["rotation_plane"]
    dt = 0.001
    steps = int(t_total / dt)

    model = create_basic_model(r)
    data = mujoco.MjData(model)

    for i in range(steps):
        angle = omega * dt * i
        if plane == "xy-plane":
            data.qpos[0] = r * np.cos(angle)  # x
            data.qpos[1] = r * np.sin(angle)  # y
            data.qpos[2] = 0.0
        elif plane == "xz-plane":
            data.qpos[0] = r * np.cos(angle)  # x
            data.qpos[1] = 0.0
            data.qpos[2] = r * np.sin(angle)  # z
        else:  # "yz-plane"
            data.qpos[0] = 0.0
            data.qpos[1] = r * np.cos(angle)  # y
            data.qpos[2] = r * np.sin(angle)  # z

        for j in range(3):
            data.qvel[j] = 0

        mujoco.mj_step(model, data)

    return {
        "x_B": round(data.qpos[0], 4),
        "y_B": round(data.qpos[1], 4),
        "z_B": round(data.qpos[2], 4)
    }


# Simulate 3D projectile motion
def simulate_3d_projectile_motion(v0, angle):
    model = create_basic_model(0)
    data = mujoco.MjData(model)

    dt = 0.001
    g = 9.81
    vz = v0[2]
    t_total = 2 * vz / g  # flight time until it hits the ground again

    for i in range(int(t_total / dt)):
        for j in range(3):
            if j == 2:  # z-axis affected by gravity
                data.qvel[j] = v0[j] - g * dt * i
                data.qpos[j] = v0[j] * dt * i - 0.5 * g * (dt * i) ** 2
            else:
                data.qvel[j] = v0[j]
                data.qpos[j] = v0[j] * dt * i
        mujoco.mj_step(model, data)

    return {
        "flight_time": round(t_total, 4),
        "maximum_height": round((vz ** 2) / (2 * g), 4),
        "range_x": round(data.qpos[0], 4),
        "range_y": round(data.qpos[1], 4),
        "range_z": round(data.qpos[2], 4)
    }


def simulate_3d_multi_object_motion(parameters):
    results = {}

    # Object A: Linear Motion (Analytical solution)
    v0 = np.array(parameters["object_A"]["v0"])
    a = np.array(parameters["object_A"]["a"])
    t = parameters["object_A"]["t"]
    pos_A = v0 * t + 0.5 * a * (t ** 2)
    results["pos_A"] = {
        "x_A": round(pos_A[0], 4),
        "y_A": round(pos_A[1], 4),
        "z_A": round(pos_A[2], 4)
    }

    # Object B: Circular Motion
    r = parameters["object_B"]["r"]
    omega = parameters["object_B"]["omega"]
    t = parameters["object_B"]["t"]
    plane = parameters["object_B"]["rotation_plane"]
    theta = omega * t

    if plane == "xy-plane":
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0.0
    elif plane == "xz-plane":
        x = r * np.cos(theta)
        y = 0.0
        z = r * np.sin(theta)
    else:  # yz-plane
        x = 0.0
        y = r * np.cos(theta)
        z = r * np.sin(theta)

    results["pos_B"] = {
        "x_B": round(x, 4),
        "y_B": round(y, 4),
        "z_B": round(z, 4)
    }

    # Object C: Projectile Motion
    v0_c = np.array(parameters["object_C"]["v0"])
    angle_deg = parameters["object_C"]["angle"]
    t = parameters["object_C"]["t"]
    g = 9.81  # gravity in z-direction

    pos_C = np.zeros(3)
    pos_C[0] = v0_c[0] * t  # x
    pos_C[1] = v0_c[1] * t  # y
    pos_C[2] = v0_c[2] * t - 0.5 * g * (t ** 2)  # z

    results["pos_C"] = {
        "x_C": round(pos_C[0], 4),
        "y_C": round(pos_C[1], 4),
        "z_C": round(pos_C[2], 4)
    }

    return results


def simulate_3d_collision(m1, m2, p1, p2, v1, v2, r, sim_steps=1000, dt=0.001):
    """
    模拟两个球体发生弹性碰撞，并返回是否碰撞以及最终速度（真实模拟+碰撞处理）。
    """
    xml = f"""
    <mujoco>
        <option timestep="{dt}"/>
        <option gravity="0 0 0"/>
        <worldbody>
            <body name="obj1" pos="{p1[0]} {p1[1]} {p1[2]}">
                <freejoint/>
                <geom type="sphere" size="{r}" mass="{m1}" rgba="1 0 0 1"/>
            </body>
            <body name="obj2" pos="{p2[0]} {p2[1]} {p2[2]}">
                <freejoint/>
                <geom type="sphere" size="{r}" mass="{m2}" rgba="0 0 1 1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    v1 = np.array(v1)
    v2 = np.array(v2)
    p1 = np.array(p1)
    p2 = np.array(p2)
    # 设置初始速度
    data.qvel[:3] = v1
    data.qvel[6:9] = v2

    collided = False
    pos1, pos2 = p1, p2

    for i in range(sim_steps):
        mujoco.mj_step(model, data)

        v1_cur = data.qvel[:3]
        v2_cur = data.qvel[6:9]
        pos1 += v1_cur * i * dt
        pos2 += v2_cur * i * dt

        dist = np.linalg.norm(pos1 - pos2)

        if not collided and dist <= 2 * r + 1e-4:
            collided = True
            n = (pos1 - pos2) / np.linalg.norm(pos1 - pos2)

            v_rel = np.dot(v1_cur - v2_cur, n)
            if v_rel > 0:  # 避免物体分离状态时触发碰撞
                continue

            # 弹性碰撞公式（在 n 方向）
            v1_new = v1_cur - (2 * m2 / (m1 + m2)) * v_rel * n
            v2_new = v2_cur + (2 * m1 / (m1 + m2)) * v_rel * n

            data.qvel[:3] = v1_new
            data.qvel[6:9] = v2_new

    # 获取最终速度
    v1_final = data.qvel[:3]
    v2_final = data.qvel[6:9]
    if collided:
        return {
            "will_collide": "true",
            "velocity_1": {
                'vel_1_x': round(float(v1_final[0]), 4),
                'vel_1_y': round(float(v1_final[1]), 4),
                'vel_1_z': round(float(v1_final[2]), 4),
            },
            "velocity_2": {
                'vel_2_x': round(float(v2_final[0]), 4),
                'vel_2_y': round(float(v2_final[1]), 4),
                'vel_2_z': round(float(v2_final[2]), 4),
            },
        }
    else:
        return {
            "will_collide": "false",
            "velocity_1": {"vel_1_x": "", "vel_1_y": "", "vel_1_z": ""},
            "velocity_2": {"vel_2_x": "", "vel_2_y": "", "vel_2_z": ""}
        }


def solve_problem(question):
    q = question
    t = q["type"]
    p = q["parameters"]
    if t == "3D Linear Motion":
        q["answer_json"] = simulate_3d_linear_motion(**p)
    elif t == "3D Circular Motion":
        q["answer_json"] = simulate_3d_circular_motion(p)
    elif t == "3D Projectile Motion":
        q["answer_json"] = simulate_3d_projectile_motion(**p)
    elif t == "3D Multi-Object Motion":
        q["answer_json"] = simulate_3d_multi_object_motion(p)
    elif t == "3D Collision":
        q["answer_json"] = simulate_3d_collision(**p)

    return q["answer_json"]

def get_all_body_states(model, data):
    states = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
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

        for action_name, action_desc in available_moves.items():
            sim_data = copy.deepcopy(env_data)

            robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            dof_start = model.body_dofadr[robot_body_id]

            # 控制逻辑
            if action_name == "move_left":
                sim_data.qvel[dof_start + 0] -= 1.0
            elif action_name == "move_right":
                sim_data.qvel[dof_start + 0] += 1.0
            elif action_name == "move_up":
                sim_data.qvel[dof_start + 1] += 1.0
            elif action_name == "move_down":
                sim_data.qvel[dof_start + 1] -= 1.0
            elif action_name == "jump":
                sim_data.qvel[dof_start + 2] += 1.0
            elif action_name == "stay":
                pass

            # 计算仿真步数
            steps = int(0.2 / model.opt.timestep) if action_name == "jump" else int(1.0 / model.opt.timestep)

            for _ in range(steps):
                mujoco.mj_step(model, sim_data)

            sim_results[action_name] = {
                "final_robot_pos": get_all_body_states(model,sim_data),
                "description": action_desc
            }

        return sim_results

    def sim(self, model, env_data, action):
        if self.method == 'mujoco':
            return self.mujoco_sim(model, env_data, action)
        return None


if __name__ == "__main__":
    with open("../../dataset/physics_questions.json", "r") as f:
        questions = json.load(f)

    for q in questions:
        t = q["type"]
        p = q["parameters"]
        if t == "3D Linear Motion":
            q["answer_json"] = simulate_3d_linear_motion(**p)
        elif t == "3D Circular Motion":
            q["answer_json"] = simulate_3d_circular_motion(p)
        elif t == "3D Projectile Motion":
            q["answer_json"] = simulate_3d_projectile_motion(**p)
        elif t == "3D Multi-Object Motion":
            q["answer_json"] = simulate_3d_multi_object_motion(p)
        elif t == "3D Collision":
            q["answer_json"] = simulate_3d_collision(**p)
    # Save output
    output_path = "../../dataset/physics_answer_sim.json"
    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)


    def compare_answers_with_tolerance(tol=0.05):
        with open("../../dataset/physics_ground_truth.json", "r") as f1:
            questions = json.load(f1)

        with open("../../dataset/physics_answer_sim.json", "r") as f2:
            answers = json.load(f2)

        assert len(questions) == len(answers), "两个文件长度不一致！"

        for i in range(len(questions)):
            q = questions[i]
            sim = answers[i]

            ans1 = q.get("answer_json", {})
            ans2 = sim.get("answer_json", {})

            # 如果结构不一样直接报错
            if set(ans1.keys()) != set(ans2.keys()):
                print(f"\n❗️[Mismatch Keys @ Question {i}]")
                print("Keys in ground_truth:", ans1.keys())
                print("Keys in simulation:", ans2.keys())
                continue

            diff = {}
            for key in ans1:
                try:
                    if isinstance(ans1[key], dict) and isinstance(ans2.get(key), dict):
                        # 递归比较子字典
                        for _key in ans1[key]:
                            v1 = float(ans1[key][_key])
                            v2 = float(ans2[key][_key])
                            if abs(v1 - v2) > tol:
                                diff[key] = (v1, v2)
                    else:
                        v1 = float(ans1[key])
                        v2 = float(ans2[key])
                        if abs(v1 - v2) > tol:
                            diff[key] = (v1, v2)
                except:
                    # 比较字符串或无法转为 float 的项
                    if ans1[key] != ans2.get(key):
                        diff[key] = (ans1[key], ans2.get(key))

            if diff:
                print(f"\n❗️[Mismatch @ Question {i}] Type: {q.get('type', 'Unknown')}")
                print("Question:\n", q.get("question", "No text"))
                print("Differences (beyond ±{:.3f}):".format(tol))
                for k, (v1, v2) in diff.items():
                    print(f" - {k}: {v1} vs {v2}")


    compare_answers_with_tolerance(tol=0.05)
