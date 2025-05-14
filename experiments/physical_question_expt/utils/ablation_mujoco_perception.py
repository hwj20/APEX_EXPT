'''
做 dt的笑容实验
'''
import json
import time

import mujoco
import numpy as np
import pandas as pd


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
def simulate_3d_linear_motion(v0, a, t, dt):
    model = create_basic_model(0)
    data = mujoco.MjData(model)

    # initialize
    vel = np.array(v0, dtype=float)  # current velocity
    pos = np.zeros(3, dtype=float)  # current position

    steps = int(t / dt)
    for _ in range(steps):
        # Euler integrate velocity and position
        vel += np.array(a, dtype=float) * dt
        pos += vel * dt

        # write back into MuJoCo state
        data.qvel[:3] = vel
        data.qpos[:3] = pos

        mujoco.mj_step(model, data)

    return {
        "velocity_x": round(data.qvel[0], 4),
        "velocity_y": round(data.qvel[1], 4),
        "velocity_z": round(data.qvel[2], 4),
        "displacement_x": round(data.qpos[0], 4),
        "displacement_y": round(data.qpos[1], 4),
        "displacement_z": round(data.qpos[2], 4)
    }


def simulate_3d_circular_motion(p, dt):
    r = p["r"]
    omega = p["omega"]
    t_total = p["t"]
    plane = p["rotation_plane"]
    steps = int(t_total / dt)

    model = create_basic_model(r)
    data = mujoco.MjData(model)

    angle = 0.0
    if plane == "xy-plane":
        pos = np.array([r, 0.0, 0.0])
    elif plane == "xz-plane":
        pos = np.array([r, 0.0, 0.0])
    else:  # "yz-plane"
        pos = np.array([0.0, r, 0.0])

    data.qpos[:3] = pos
    data.qvel[:3] = 0.0

    for _ in range(steps):
        angle += omega * dt

        if plane == "xy-plane":
            new_pos = np.array([r * np.cos(angle),
                                r * np.sin(angle),
                                0.0])
        elif plane == "xz-plane":
            new_pos = np.array([r * np.cos(angle),
                                0.0,
                                r * np.sin(angle)])
        else:  # "yz-plane"
            new_pos = np.array([0.0,
                                r * np.cos(angle),
                                r * np.sin(angle)])

        vel = (new_pos - pos) / dt

        data.qpos[:3] = new_pos
        data.qvel[:3] = vel

        mujoco.mj_step(model, data)

        pos = new_pos

    return {
        "x_B": round(data.qpos[0], 4),
        "y_B": round(data.qpos[1], 4),
        "z_B": round(data.qpos[2], 4)
    }


# Simulate 3D projectile motion
def simulate_3d_projectile_motion(v0, angle_unused, dt):
    """
    Euler-step simulation of 3D projectile motion.
    v0: initial velocity vector [vx, vy, vz]
    angle_unused: not needed, since v0 already has components
    dt: timestep
    """
    model = create_basic_model(0)
    data = mujoco.MjData(model)

    vel = np.array(v0, dtype=float)
    pos = np.zeros(3, dtype=float)

    g = 9.81
    flight_time = 2 * v0[2] / g
    steps = int(flight_time / dt)

    for _ in range(steps):
        vel[2] -= g * dt

        pos += vel * dt
        if pos[2] <= 0:
            break

        data.qvel[:3] = vel
        data.qpos[:3] = pos

        mujoco.mj_step(model, data)

    max_height = (v0[2] ** 2) / (2 * g)

    return {
        "flight_time": round(flight_time, 4),
        "maximum_height": round(max_height, 4),
        "range_x": round(data.qpos[0], 4),
        "range_y": round(data.qpos[1], 4),
        "range_z": round(data.qpos[2], 4)
    }


def simulate_3d_projectile_position(v0, t, dt):
    model = create_basic_model(0)
    data = mujoco.MjData(model)

    vel = np.array(v0, dtype=float)
    pos = np.zeros(3, dtype=float)
    g = 9.81

    steps = int(t / dt)
    for _ in range(steps):
        vel[2] -= g * dt

        pos += vel * dt

        data.qvel[:3] = vel
        data.qpos[:3] = pos

        mujoco.mj_step(model, data)

    return {
        "x_C": round(data.qpos[0], 4),
        "y_C": round(data.qpos[1], 4),
        "z_C": round(data.qpos[2], 4)
    }


def simulate_3d_multi_object_motion(parameters, dt):
    """
    Simulate multi-object motion using individual simulation functions for each object.
    """
    # Object A: Linear Motion via simulation
    linear_res = simulate_3d_linear_motion(
        np.array(parameters["object_A"]["v0"]),
        np.array(parameters["object_A"]["a"]),
        parameters["object_A"]["t"],
        dt
    )
    # Rename keys to match multi-object format
    pos_A = {
        "x_A": linear_res["displacement_x"],
        "y_A": linear_res["displacement_y"],
        "z_A": linear_res["displacement_z"]
    }

    # Object B: Circular Motion via simulation
    circ_res = simulate_3d_circular_motion(parameters["object_B"], dt)
    pos_B = {
        "x_B": circ_res["x_B"],
        "y_B": circ_res["y_B"],
        "z_B": circ_res["z_B"]
    }

    # Object C: Projectile Motion via simulation
    v0_c = np.array(parameters["object_C"]["v0"])
    t_c = parameters["object_C"]["t"]
    pos_C_sim = simulate_3d_projectile_position(v0_c, t_c, dt)
    pos_C = {
        "x_C": pos_C_sim["x_C"],
        "y_C": pos_C_sim["y_C"],
        "z_C": pos_C_sim["z_C"]
    }

    return {"pos_A": pos_A, "pos_B": pos_B, "pos_C": pos_C}


def simulate_3d_collision(m1, m2, p1, p2, v1, v2, r, sim_steps=1000, dt=0.01):
    """ elastic collision """
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
    # initial velocity
    data.qvel[:3] = v1
    data.qvel[6:9] = v2

    collided = False
    pos1, pos2 = p1, p2

    for i in range(sim_steps):
        mujoco.mj_step(model, data)

        v1_cur = data.qvel[:3]
        v2_cur = data.qvel[6:9]
        pos1 += v1_cur * dt
        pos2 += v2_cur * dt

        dist = np.linalg.norm(pos1 - pos2)

        if not collided and dist <= 2 * r + 1e-4:
            collided = True
            n = (pos1 - pos2) / np.linalg.norm(pos1 - pos2)

            v_rel = np.dot(v1_cur - v2_cur, n)
            if v_rel > 0:
                continue

            # elastic collision
            v1_new = v1_cur - (2 * m2 / (m1 + m2)) * v_rel * n
            v2_new = v2_cur + (2 * m1 / (m1 + m2)) * v_rel * n

            data.qvel[:3] = v1_new
            data.qvel[6:9] = v2_new

    # final velocity
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


# just to compare the sim results with ground truth (without LLM Answering)
def compare_answers_with_tolerance(file_name, tol=0.05):
    with open("../dataset/physics_ground_truth.json", "r") as f1:
        questions = json.load(f1)

    with open(file_name, "r") as f2:
        answers = json.load(f2)

    assert len(questions) == len(answers), "The number of questions are different"

    # Initialize statistics for each question type
    stats = {}
    for i in range(len(questions)):
        q = questions[i]
        ans_true = q.get("answer_json", {})
        ans_pred = answers[i].get("answer_json", {})

        q_type = q.get("type", "Unknown")
        if q_type not in stats:
            stats[q_type] = {"correct": 0, "total": 0}
        stats[q_type]["total"] += 1

        # Check structural mismatch
        if set(ans_true.keys()) != set(ans_pred.keys()):
            print(f"\n❗️[Mismatch Keys @ Q{i}] Type: {q_type}")
            print("Keys in ground truth:", ans_true.keys())
            print("Keys in prediction:", ans_pred.keys())
            continue

        # Compare values
        diff_found = False
        for key in ans_true:
            if isinstance(ans_true[key], dict) and isinstance(ans_pred[key], dict):
                for subkey in ans_true[key]:
                    try:
                        v1 = float(ans_true[key][subkey])
                        v2 = float(ans_pred[key][subkey])
                        if abs(v1 - v2) > max(0.05, abs(tol * v1)):
                            diff_found = True
                            break
                    except:
                        if ans_true[key][subkey] != ans_pred[key].get(subkey):
                            diff_found = True
                            break
            else:
                try:
                    v1 = float(ans_true[key])
                    v2 = float(ans_pred.get(key, ""))
                    if abs(v1 - v2) > max(0.05, abs(tol * v1)):
                        diff_found = True
                except:
                    if ans_true[key] != ans_pred.get(key):
                        diff_found = True
            if diff_found:
                break

        if not diff_found:
            stats[q_type]["correct"] += 1
        # else:
        # print(f"\n❗️[Mismatch Keys @ Q{i}]: {q_type}")
        # print("Keys in ground truth:", ans_true)
        # print("Keys in prediction:", ans_pred)

    # Print accuracy per question type
    print("\nAccuracy per question type:")
    for q_type, counts in stats.items():
        correct = counts["correct"]
        total = counts["total"]
        accuracy = (correct / total) * 100
        print(f" - {q_type}: {accuracy:.2f}% ({correct}/{total})")


def solve_problem(question, dt):
    t = question["type"]
    p = question["parameters"]
    if t == "3D Linear Motion":
        return simulate_3d_linear_motion(np.array(p["v0"]), np.array(p["a"]), p["t"], dt)
    elif t == "3D Circular Motion":
        return simulate_3d_circular_motion(p, dt)
    elif t == "3D Projectile Motion":
        return simulate_3d_projectile_motion(np.array(p["v0"]), p["angle"], dt)
    elif t == "3D Multi-Object Motion":
        return simulate_3d_multi_object_motion(p, dt)
    elif t == "3D Collision":
        return simulate_3d_collision(
            p["m1"], p["m2"], p["p1"], p["p2"], p["v1"], p["v2"], p["r"], sim_steps=1000, dt=dt
        )


if __name__ == "__main__":
    # Load the questions
    with open("../dataset/physics_questions.json", "r") as f:
        questions = json.load(f)

    dt_values = [0.001, 0.005, 0.01]

    for dt in dt_values:
        records = []
        results = []

        # Process each question
        for q in questions:
            start = time.time()
            answer = solve_problem(q, dt)
            elapsed = time.time() - start

            # Record timing
            records.append({
                "dt": dt,
                "question_type": q["type"],
                "duration_s": elapsed
            })

            # Save answer
            q_with_ans = q.copy()
            q_with_ans["answer_json"] = answer
            results.append(q_with_ans)

        # Save results for this dt
        out_path = f"../dataset/physics_answer_sim_dt_{dt}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for dt={dt} to {out_path}")

        # Compare answers with tolerance and print accuracy
        print(f"\nAccuracy for dt={dt}:")
        compare_answers_with_tolerance(out_path, tol=0.05)

        # Compute and display timing statistics
        df = pd.DataFrame(records)
        df_avg = df.groupby(["dt", "question_type"])["duration_s"].mean().reset_index()
        print(f"\nAverage duration per question type for dt={dt}:")
        print(df_avg[df_avg["dt"] == dt])
