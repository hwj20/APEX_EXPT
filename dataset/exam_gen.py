import json
import random
import math

random.seed(42)
# 生成直线运动问题
def generate_linear_motion():
    v0 = round(random.uniform(1, 20), 2)  # 初速度 m/s
    a = round(random.uniform(-5, 5), 2)  # 加速度 m/s²
    t = round(random.uniform(1, 10), 2)  # 时间 s
    return {
        "type": "Linear Motion",
        "question": f"An object starts with an initial velocity of {v0} m/s and an acceleration of {a} m/s². What is its velocity and displacement after {t} seconds?",
        "parameters": {"v0": v0, "a": a, "t": t}
    }

# 生成圆周运动问题
def generate_circular_motion():
    r = round(random.uniform(0.5, 5), 2)  # 半径 m
    v = round(random.uniform(1, 10), 2)  # 线速度 m/s
    omega = round(v / r, 2)  # 角速度 rad/s
    return {
        "type": "Circular Motion",
        "question": f"An object moves in a circular path with a radius of {r} meters and a speed of {v} m/s. What is its angular velocity and centripetal acceleration?",
        "parameters": {"r": r, "v": v, "omega": omega}
    }

# 生成抛物运动问题
def generate_projectile_motion():
    v0 = round(random.uniform(5, 30), 2)  # 初速度 m/s
    angle = round(random.uniform(15, 75), 2)  # 发射角度 degree
    return {
        "type": "Projectile Motion",
        "question": f"A projectile is launched with an initial speed of {v0} m/s at an angle of {angle} degrees. Calculate its flight time, maximum height, and range.",
        "parameters": {"v0": v0, "angle": angle}
    }

# 生成多物体运动问题（直线、圆周、抛物运动的坐标计算）
def generate_multi_object_motion():
    # 直线运动物体 A
    v_A = round(random.uniform(1, 20), 2)  # m/s
    a_A = round(random.uniform(-5, 5), 2)  # m/s²
    t_A = round(random.uniform(1, 10), 2)  # s

    # 圆周运动物体 B
    r_B = round(random.uniform(0.5, 5), 2)  # m
    v_B = round(random.uniform(1, 10), 2)  # m/s
    omega_B = round(v_B / r_B, 2)  # 角速度 rad/s
    t_B = round(random.uniform(1, 10), 2)  # s

    # 抛物运动物体 C
    v_C = round(random.uniform(5, 30), 2)  # m/s
    theta_C = round(random.uniform(15, 75), 2)  # degrees
    t_C = round(random.uniform(1, 10), 2)  # s

    return {
        "type": "Multi-Object Motion",
        "question": f"""
Three objects move in different types of motion:
1. **Object A (Linear Motion)**  
   - Initial velocity: {v_A} m/s  
   - Acceleration: {a_A} m/s²  
   - Time: {t_A} s  
   - Compute its final position (x_A, y_A). Assume it starts at (0, 0).

2. **Object B (Circular Motion)**  
   - Radius: {r_B} meters  
   - Speed: {v_B} m/s  
   - Angular velocity: {omega_B} rad/s  
   - Time: {t_B} s  
   - Compute its position (x_B, y_B) assuming it starts at (r_B, 0).

3. **Object C (Projectile Motion)**  
   - Initial speed: {v_C} m/s  
   - Launch angle: {theta_C} degrees  
   - Time: {t_C} s  
   - Compute its position (x_C, y_C), assuming it starts from (0,0).
""",
        "parameters": {
            "object_A": {"v0": v_A, "a": a_A, "t": t_A},
            "object_B": {"r": r_B, "v": v_B, "omega": omega_B, "t": t_B},
            "object_C": {"v0": v_C, "angle": theta_C, "t": t_C}
        }
    }

# 生成碰撞问题（弹性碰撞）
def generate_collision_problem():
    m1 = round(random.uniform(1, 10), 2)  # 质量 kg
    m2 = round(random.uniform(1, 10), 2)  # 质量 kg
    v1 = round(random.uniform(-10, 10), 2)  # 初速度 m/s
    v2 = round(random.uniform(-10, 10), 2)  # 初速度 m/s

    return {
        "type": "Collision",
        "question": f"Two objects with masses {m1} kg and {m2} kg are moving with velocities {v1} m/s and {v2} m/s. Assuming an elastic collision, what are their final velocities?",
        "parameters": {"m1": m1, "m2": m2, "v1": v1, "v2": v2}
    }

# 重新生成包含 50 道碰撞题的完整物理题集
def generate_full_physics_questions():
    questions = []
    for _ in range(25):
        questions.append(generate_linear_motion())
    for _ in range(25):
        questions.append(generate_circular_motion())
    for _ in range(25):
        questions.append(generate_projectile_motion())
    for _ in range(75):
        questions.append(generate_multi_object_motion())
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
