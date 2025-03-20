import numpy as np

def solve_linear_motion(v0, a, t):
    """计算匀加速直线运动的最终速度和位移"""
    v = v0 + a * t
    s = v0 * t + 0.5 * a * (t ** 2)
    return {"velocity": v, "displacement": s}

# 示例：计算一题的正确答案
params = {
    "v0": 1.5,  # 初速度
    "a": -3.01, # 加速度
    "t": 6.85   # 时间
}

answer = solve_linear_motion(**params)
print(f"正确答案: 速度 = {answer['velocity']} m/s, 位移 = {answer['displacement']} m")
