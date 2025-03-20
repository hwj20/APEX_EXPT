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
    """ 生成 3D 圆周运动问题 """

    # 圆周半径 m
    r = round(random.uniform(0.5, 5), 2)
    # 线速度 (vx, vy, vz) m/s
    v = np.array([round(random.uniform(1, 10), 2) for _ in range(3)])
    # 角速度 rad/s
    omega = round(np.linalg.norm(v) / r, 2)
    # 旋转平面
    rotation_plane = random.choice(["xy-plane", "xz-plane", "yz-plane"])

    return {
        "type": "3D Circular Motion",
        "question": f"""
An object moves in a circular path of radius {r} meters at a speed of {v.tolist()} m/s in the {rotation_plane}.
Calculate its angular velocity (ω) and centripetal acceleration (ax, ay, az).
""",
        "parameters": {"r": r, "v": v.tolist(), "omega": omega, "rotation_plane": rotation_plane},
        "answer_json": {
            "angular_velocity": "",
            "centripetal_acceleration_x": "", "centripetal_acceleration_y": "", "centripetal_acceleration_z": ""
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
    v_C = np.array([round(random.uniform(5, 30), 2), round(random.uniform(5, 30), 2), round(random.uniform(5, 30), 2)])  # 初速度 (vx, vy, vz) m/s
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



# 生成碰撞问题（弹性碰撞）
def generate_collision_problem():
    """ 生成 3D 碰撞问题，判断是否碰撞 & 计算碰撞后轨迹 """

    # 质量（kg）
    m1 = round(random.uniform(1, 10), 2)
    m2 = round(random.uniform(1, 10), 2)

    # 初始位置 (x, y, z)
    p1 = np.array([round(random.uniform(-5, 5), 2) for _ in range(3)])
    p2 = np.array([round(random.uniform(-5, 5), 2) for _ in range(3)])

    # 初速度 (vx, vy, vz)
    v1 = np.array([round(random.uniform(-10, 10), 2) for _ in range(3)])
    v2 = np.array([round(random.uniform(-10, 10), 2) for _ in range(3)])


    return {
        "type": "3D Collision",
        "question": (
            f"Two objects with masses {m1} kg and {m2} kg are located at positions {p1} and {p2}, "
            f"moving with velocities {v1} m/s and {v2} m/s. Assuming an elastic collision, will they collide? "
            f"If they collide, what are their final velocities?"
        ),
        "parameters": {"m1": m1, "m2": m2, "p1": p1.tolist(), "p2": p2.tolist(), "v1": v1.tolist(), "v2": v2.tolist()},
        "answer_json": {
            "will_collide": "true or false",
            "velocity_1": {"vel_1_x":"","vel_1_y":"","vel_1_z":""},
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
save_full_questions_to_json()
