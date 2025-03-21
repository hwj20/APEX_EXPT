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
    v = np.array(p["v"])
    omega = np.linalg.norm(v) / r

    acc = -(v ** 2) / r  # Simple assumption: centripetal acc per axis
    if p["rotation_plane"] == "xy-plane":
        acc = np.array([-v[0] * omega, -v[1] * omega, 0])
    elif p["rotation_plane"] == "xz-plane":
        acc = np.array([-v[0] * omega, 0, -v[2] * omega])
    else:
        acc = np.array([0, -v[1] * omega, -v[2] * omega])

    return {
        "angular_velocity": round(omega, 2),
        "centripetal_acceleration_x": round(acc[0], 2),
        "centripetal_acceleration_y": round(acc[1], 2),
        "centripetal_acceleration_z": round(acc[2], 2),
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


def solve_collision(p):
    m1, m2 = p["m1"], p["m2"]
    p1, p2 = np.array(p["p1"]), np.array(p["p2"])
    v1, v2 = np.array(p["v1"]), np.array(p["v2"])
    rel_pos = p2 - p1
    rel_vel = v2 - v1
    if np.dot(rel_pos, rel_vel) >= 0:
        will_collide = False
    else:
        will_collide = True

    if will_collide:
        v1_new = v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, p1 - p2) / np.linalg.norm(p1 - p2) ** 2 * (p1 - p2)
        v2_new = v2 - (2 * m1 / (m1 + m2)) * np.dot(v2 - v1, p2 - p1) / np.linalg.norm(p2 - p1) ** 2 * (p2 - p1)
        return {
            "will_collide": "true",
            "velocity_1": {"vel_1_x": round(v1_new[0], 2), "vel_1_y": round(v1_new[1], 2),
                           "vel_1_z": round(v1_new[2], 2)},
            "velocity_2": {"vel_2_x": round(v2_new[0], 2), "vel_2_y": round(v2_new[1], 2),
                           "vel_2_z": round(v2_new[2], 2)}
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
    elif t == "3D Collision":
        q["answer_json"] = solve_collision(p)

# Save output
output_path = "../dataset/physics_ground_truth.json"
with open(output_path, "w") as f:
    json.dump(questions, f, indent=2)

output_path
