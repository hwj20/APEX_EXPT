import json
import random
import math
import numpy as np

random.seed(42)


def generate_3d_linear_motion():
    """ 生成 3D 直线运动问题 """

    # 初速度 (vx, vy, vz) m/s
    v0 = np.array([round(random.uniform(1, 20), 2) for _ in range(3)])
    # 加速度 (ax, ay, az) m/s²
    a = np.array([round(random.uniform(-5, 5), 2) for _ in range(3)])
    # 时间 s
    t = round(random.uniform(1, 10), 2)

    return {
        "type": "3D Linear Motion",
        "question": f"""
An object starts at (0,0,0) with an initial velocity of {v0.tolist()} m/s and an acceleration of {a.tolist()} m/s².
What are its velocity (vx, vy, vz) and displacement (dx, dy, dz) after {t} seconds?
""",
        "parameters": {"v0": v0.tolist(), "a": a.tolist(), "t": t},
        "answer_json": {
            "velocity_x": "", "velocity_y": "", "velocity_z": "",
            "displacement_x": "", "displacement_y": "", "displacement_z": ""
        }
    }


def generate_3d_circular_motion():
    """ 生成更合理的 3D 圆周运动问题（基于标量速度 & 明确平面） """
    r = round(random.uniform(0.5, 5), 2)  # 半径
    v = round(random.uniform(1, 10), 2)   # 标量速度
    omega = round(v / r, 2)               # 角速度
    t = round(random.uniform(1, 10), 2)   # 时间
    plane = random.choice(["xy-plane", "xz-plane", "yz-plane"])

    return {
        "type": "3D Circular Motion",
        "question": f"""
        Object  (Circular Motion)  
           - Radius: {r} meters  
           - Speed: {v} m/s  
           - Angular velocity: {omega} rad/s  
           - Time: {t} s  
           - Rotating in the {plane}  
           - Compute its position (x_B, y_B, z_B), assuming it starts at (r, 0, 0).
        """,
        "parameters": {
            "r": r,
            "v": v,
            "omega": omega,
            "t": t,
            "rotation_plane": plane
        },
        "answer_json": {
            "x_B": "",
            "y_B": "",
            "z_B": ""
        }
    }


def generate_3d_projectile_motion():
    """ 生成 3D 抛物运动问题 """

    # 初速度 (vx, vy, vz) m/s
    v0 = np.array([round(random.uniform(5, 30), 2) for _ in range(3)])
    # 发射角度（相对于 x-y 平面）
    angle = round(random.uniform(15, 75), 2)

    return {
        "type": "3D Projectile Motion",
        "question": f"""
            A projectile is launched from (0,0,0) with an initial velocity of {v0.tolist()} m/s at an angle of {angle} degrees.
            Calculate its flight time, maximum height (h), and range (dx, dy, dz).
            """,
        "parameters": {"v0": v0.tolist(), "angle": angle},
        "answer_json": {
            "flight_time": "",
            "maximum_height": "",
            "range_x": "", "range_y": "", "range_z": ""
        }
    }


# 生成多物体运动问题（直线、圆周、抛物运动的坐标计算）
def generate_3d_multi_object_motion():
    """ 生成 3D 版本的多物体运动问题 """

    # 直线运动物体 A（随机方向）
    v_A = np.array([round(random.uniform(1, 20), 2) for _ in range(3)])  # 初速度 (vx, vy, vz) m/s
    a_A = np.array([round(random.uniform(-5, 5), 2) for _ in range(3)])  # 加速度 (ax, ay, az) m/s²
    t_A = round(random.uniform(1, 10), 2)  # 时间 s

    # 圆周运动物体 B（随机选择旋转轴）
    r_B = round(random.uniform(0.5, 5), 2)  # 轨道半径 m
    v_B = round(random.uniform(1, 10), 2)  # 线速度 m/s
    omega_B = round(v_B / r_B, 2)  # 角速度 rad/s
    t_B = round(random.uniform(1, 10), 2)  # 时间 s
    rotation_axis = random.choice(["xy-plane", "xz-plane", "yz-plane"])  # 选择旋转平面

    # 抛物运动物体 C（3D 抛出）
    v_C = np.array([round(random.uniform(5, 30), 2), round(random.uniform(5, 30), 2),
                    round(random.uniform(5, 30), 2)])  # 初速度 (vx, vy, vz) m/s
    theta_C = round(random.uniform(15, 75), 2)  # 发射角度（相对地面）
    t_C = round(random.uniform(1, 10), 2)  # 时间 s

    return {
        "type": "3D Multi-Object Motion",
        "question": f"""
            Three objects move in different types of motion in 3D space:
            1. Object A (Linear Motion)  
               - Initial velocity: {v_A.tolist()} m/s  
               - Acceleration: {a_A.tolist()} m/s²  
               - Time: {t_A} s  
               - Compute its final position (x_A, y_A, z_A). Assume it starts at (0, 0, 0).

            2. Object B (Circular Motion)  
               - Radius: {r_B} meters  
               - Speed: {v_B} m/s  
               - Angular velocity: {omega_B} rad/s  
               - Time: {t_B} s  
               - Rotating in the {rotation_axis}  
               - Compute its position (x_B, y_B, z_B), assuming it starts at (r_B, 0, 0).

            3. Object C (Projectile Motion)  
               - Initial speed: {v_C.tolist()} m/s  
               - Launch angle: {theta_C} degrees  
               - Time: {t_C} s  
               - Compute its position (x_C, y_C, z_C), assuming it starts from (0,0,0).
            """,
        "parameters": {
            "object_A": {"v0": v_A.tolist(), "a": a_A.tolist(), "t": t_A},
            "object_B": {"r": r_B, "v": v_B, "omega": omega_B, "t": t_B, "rotation_plane": rotation_axis},
            "object_C": {"v0": v_C.tolist(), "angle": theta_C, "t": t_C}
        },
        "answer_json": {
            "pos_A": {"x_A": "", "y_A": "", "z_A": ""},
            "pos_B": {"x_B": "", "y_B": "", "z_B": ""},
            "pos_C": {"x_C": "", "y_C": "", "z_C": ""}
        }
    }


def generate_collision_problem(bias_ratio=0.5):
    """
    生成更合理的 3D 碰撞问题，bias_ratio 决定撞击题目的比例。
    """
    # 质量（kg）
    m1 = round(random.uniform(1, 10), 2)
    m2 = round(random.uniform(1, 10), 2)
    r = 0.5  # 半径

    will_collide = random.random() < bias_ratio

    if will_collide:
        # 生成一对即将相撞的物体（位置相距不远，速度相对靠近）
        center = np.array([random.uniform(-2, 2) for _ in range(3)])
        offset = np.array([random.uniform(1.5, 3) for _ in range(3)])
        p1 = center - offset / 2
        p2 = center + offset / 2

        # 相对速度向彼此靠近
        v_direction = p1 - p2
        v_direction = v_direction / np.linalg.norm(v_direction)
        speed1 = random.uniform(2, 6)
        speed2 = random.uniform(2, 6)

        v1 = -v_direction * speed1
        v2 = v_direction * speed2

    else:
        # 自由飞行，不会撞
        p1 = np.array([random.uniform(-5, 0) for _ in range(3)])
        p2 = np.array([random.uniform(0, 5) for _ in range(3)])

        # 平行方向或彼此远离
        direction = np.array([random.uniform(0.5, 1.5) for _ in range(3)])
        v1 = direction * random.uniform(2, 5)
        v2 = direction * random.uniform(2, 5)

    return {
        "type": "3D Collision",
        "question": (
            f"Two objects with masses {m1} kg and {m2} kg are located at positions {p1.tolist()} and {p2.tolist()}, "
            f"the radius of both spheres is {r}. "
            f"They are moving with velocities {v1.tolist()} m/s and {v2.tolist()} m/s. "
            f"Assuming an elastic collision, will they collide? If they collide, what are their final velocities?"
        ),
        "parameters": {
            "m1": m1, "m2": m2,
            "p1": p1.tolist(), "p2": p2.tolist(),
            "v1": v1.tolist(), "v2": v2.tolist(),
            "r": r,
            # "will_col":will_collide
        },
        "answer_json": {
            "will_collide": "true or false",
            "velocity_1": {"vel_1_x": "", "vel_1_y": "", "vel_1_z": ""},
            "velocity_2": {"vel_2_x": "", "vel_2_y": "", "vel_2_z": ""}
        }
    }



# 重新生成包含 50 道碰撞题的完整物理题集
def generate_full_physics_questions():
    questions = []
    for _ in range(25):
        questions.append(generate_3d_linear_motion())
    for _ in range(25):
        questions.append(generate_3d_circular_motion())
    for _ in range(25):
        questions.append(generate_3d_projectile_motion())
    for _ in range(75):
        questions.append(generate_3d_multi_object_motion())
    for _ in range(50):
        questions.append(generate_collision_problem())
    return questions


# 保存完整问题集到 JSON 文件
def save_full_questions_to_json():
    questions = generate_full_physics_questions()
    file_path = "physics_questions.json"
    with open(file_path, "w") as f:
        json.dump(questions, f, indent=4)
    return file_path


# 运行并保存完整文件
# save_full_questions_to_json()

def revise_json():
    file_path = "physics_questions.json"

    with open(file_path, "r") as f:
        questions = json.load(f)
    ques = []
    for q in questions:
        if q["type"] == "3D Collision":
            q = generate_collision_problem()
        ques.append(q)

    # Save revised version
    revised_path = "physics_questions.json"
    with open(revised_path, "w") as f:
        json.dump(ques, f, indent=2)


revise_json()
