import json
import numpy as np


# Physics computation functions

def solve_3d_linear_motion(p):
    v0 = np.array(p["v0"])
    a = np.array(p["a"])
    t = p["t"]

    v = v0 + a * t
    d = v0 * t + 0.5 * a * t ** 2

    return {
        "velocity_x": round(v[0], 2), "velocity_y": round(v[1], 2), "velocity_z": round(v[2], 2),
        "displacement_x": round(d[0], 2), "displacement_y": round(d[1], 2), "displacement_z": round(d[2], 2)
    }


def solve_3d_circular_motion(p):
    r = p["r"]
    omega = p["omega"]
    t = p["t"]
    plane = p["rotation_plane"]

    theta = omega * t  # 总旋转角度（rad）

    # 默认起点 (r, 0, 0)
    if plane == "xy-plane":
        x = round(r * np.cos(theta), 4)
        y = round(r * np.sin(theta), 4)
        z = 0.0
    elif plane == "xz-plane":
        x = round(r * np.cos(theta), 4)
        y = 0.0
        z = round(r * np.sin(theta), 4)
    else:  # "yz-plane"
        x = 0.0
        y = round(r * np.cos(theta), 4)
        z = round(r * np.sin(theta), 4)

    return {
        "x_B": x,
        "y_B": y,
        "z_B": z
    }

def solve_3d_projectile_motion(p):
    v0 = np.array(p["v0"])
    g = 9.81

    vz = v0[2]
    t_total = 2 * vz / g
    h = (vz ** 2) / (2 * g)
    dx = v0[0] * t_total
    dy = v0[1] * t_total
    dz = 0

    return {
        "flight_time": round(t_total, 2),
        "maximum_height": round(h, 2),
        "range_x": round(dx, 2),
        "range_y": round(dy, 2),
        "range_z": round(dz, 2)
    }


def solve_3d_multi_object_motion(params):
    # Object A (linear motion)
    v0_A = np.array(params["object_A"]["v0"])
    a_A = np.array(params["object_A"]["a"])
    t_A = params["object_A"]["t"]
    pos_A = v0_A * t_A + 0.5 * a_A * t_A ** 2

    # Object B (circular motion)
    r_B = params["object_B"]["r"]
    omega_B = params["object_B"]["omega"]
    t_B = params["object_B"]["t"]
    plane = params["object_B"]["rotation_plane"]
    theta = omega_B * t_B
    if plane == "xy-plane":
        x_B = r_B * np.cos(theta)
        y_B = r_B * np.sin(theta)
        z_B = 0
    elif plane == "xz-plane":
        x_B = r_B * np.cos(theta)
        y_B = 0
        z_B = r_B * np.sin(theta)
    else:
        x_B = 0
        y_B = r_B * np.cos(theta)
        z_B = r_B * np.sin(theta)

    # Object C (projectile motion)
    v0_C = np.array(params["object_C"]["v0"])
    t_C = params["object_C"]["t"]
    g = 9.81
    x_C = v0_C[0] * t_C
    y_C = v0_C[1] * t_C
    z_C = v0_C[2] * t_C - 0.5 * g * t_C ** 2

    return {
        "pos_A": {"x_A": round(pos_A[0], 2), "y_A": round(pos_A[1], 2), "z_A": round(pos_A[2], 2)},
        "pos_B": {"x_B": round(x_B, 2), "y_B": round(y_B, 2), "z_B": round(z_B, 2)},
        "pos_C": {"x_C": round(x_C, 2), "y_C": round(y_C, 2), "z_C": round(z_C, 2)}
    }


def solve_3d_collision(params):
    m1, m2 = params["m1"], params["m2"]
    p1, p2 = np.array(params["p1"]), np.array(params["p2"])
    v1, v2 = np.array(params["v1"]), np.array(params["v2"])
    rel_pos = p2 - p1
    rel_vel = v2 - v1
    will_collide = np.dot(rel_pos, rel_vel) < 0

    if will_collide:
        n = (p1 - p2) / np.linalg.norm(p1 - p2)
        v1_proj = np.dot(v1, n)
        v2_proj = np.dot(v2, n)
        v1_new = v1 - v1_proj * n + ((m1 - m2) * v1_proj + 2 * m2 * v2_proj) / (m1 + m2) * n
        v2_new = v2 - v2_proj * n + ((m2 - m1) * v2_proj + 2 * m1 * v1_proj) / (m1 + m2) * n
        return {
            "will_collide": "true",
            "velocity_1": {
                "vel_1_x": round(v1_new[0], 2),
                "vel_1_y": round(v1_new[1], 2),
                "vel_1_z": round(v1_new[2], 2),
            },
            "velocity_2": {
                "vel_2_x": round(v2_new[0], 2),
                "vel_2_y": round(v2_new[1], 2),
                "vel_2_z": round(v2_new[2], 2),
            }
        }
    else:
        return {
            "will_collide": "false",
            "velocity_1": {"vel_1_x": "", "vel_1_y": "", "vel_1_z": ""},
            "velocity_2": {"vel_2_x": "", "vel_2_y": "", "vel_2_z": ""}
        }


# Load questions and calculate answers
with open("../dataset/physics_questions.json", "r") as f:
    questions = json.load(f)

for q in questions:
    t = q["type"]
    p = q["parameters"]
    if t == "3D Linear Motion":
        q["answer_json"] = solve_3d_linear_motion(p)
    elif t == "3D Circular Motion":
        q["answer_json"] = solve_3d_circular_motion(p)
    elif t == "3D Projectile Motion":
        q["answer_json"] = solve_3d_projectile_motion(p)
    elif t == "3D Multi-Object Motion":
        q["answer_json"] = solve_3d_multi_object_motion(p)
    elif t == "3D Collision":
        q["answer_json"] = solve_3d_collision(p)

# Save output
output_path = "../dataset/physics_ground_truth.json"
with open(output_path, "w") as f:
    json.dump(questions, f, indent=2)

