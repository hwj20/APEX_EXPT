import mujoco
import numpy as np

def simulate_linear_motion_mujoco(v0, a, t):
    """用 MuJoCo 物理引擎模拟匀加速运动"""

    # 创建一个简单的 MuJoCo 物理环境
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
        <worldbody>
            <body name="ball" pos="0 0 0">
                <freejoint/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)

    # 设置初始速度
    data.qvel[0] = v0  # 设置 x 方向的初速度

    # 运行 MuJoCo 模拟
    dt = 0.001  # 时间步长
    steps = int(t / dt)
    for i in range(steps):
        # data.qvel[0] += a * dt  # 直接用 dv = a * dt 更新速度
        # data.qpos[0] += data.qvel[0] * dt  # 更新位置
        data.qvel[0] = a * dt * i + v0  # 直接用 dv = a * dt 更新速度
        data.qpos[0] = v0 * dt * i + 1/2*a*((dt*i)**2)  # 更新位置 x = v*t+1/2a*t**2
        mujoco.mj_step(model, data)

    # 获取最终速度和位移
    final_velocity = data.qvel[0]
    final_position = data.qpos[0]

    return {"velocity": final_velocity, "displacement": final_position}

# 示例：用 MuJoCo 真实模拟一个物体的匀加速运动
params = {
    "v0": 1.5,  # 初速度
    "a": -3.01, # 加速度
    "t": 6.85   # 运动时间
}

simulated_result = simulate_linear_motion_mujoco(**params)
print(f"MuJoCo 物理模拟结果: 速度 = {simulated_result['velocity']} m/s, 位移 = {simulated_result['displacement']} m")
