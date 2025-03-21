import mujoco
import mujoco.viewer
import numpy as np
import cv2

# ✅ 创建 MuJoCo 物理环境
model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <body name="robot" pos="0 0 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="0 1 0 1"/>
        </body>
        <body name="cat1" pos="1 1 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
        <body name="cat2" pos="-1 -1 0">
            <freejoint/>
            <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# ✅ 视频设置
fps = 30
width, height = 640, 480
video_writer = cv2.VideoWriter("turing_cat_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ✅ 初始速度设置
robot_speed = 0.02
cats = [
    {"id": 1, "vx": 0.03, "vy": 0.02},
    {"id": 2, "vx": -0.02, "vy": 0.04}
]

# ✅ 模拟循环
for step in range(300):  # 10 秒
    # 模拟机器人移动
    data.qpos[0] += robot_speed  # robot x

    # 模拟猫猫移动
    for cat in cats:
        cat_id = cat["id"]
        data.qpos[cat_id * 2] += cat["vx"]
        data.qpos[cat_id * 2 + 1] += cat["vy"]

    # PGD 预测未来位置
    future_positions = []
    for cat in cats:
        fx, fy = data.qpos[cat["id"] * 2], data.qpos[cat["id"] * 2 + 1]
        for _ in range(10):
            fx += cat["vx"]
            fy += cat["vy"]
        future_positions.append((fx, fy))

    # 如果未来要撞猫，就反向走
    for fx, fy in future_positions:
        if abs(fx - data.qpos[0]) < 0.1 and abs(fy - data.qpos[1]) < 0.1:
            robot_speed = -robot_speed

    # 模拟一步
    mujoco.mj_step(model, data)

    # 渲染并保存帧
    renderer.update_scene(data)
    pixels = renderer.render()
    frame = cv2.cvtColor(np.flipud(pixels), cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame, (width, height))
    video_writer.write(frame_resized)

# ✅ 完成录制
video_writer.release()
print("视频已保存：turing_cat_test.mp4 🎥")
