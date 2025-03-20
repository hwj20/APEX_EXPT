import mujoco
import mujoco.viewer
import numpy as np
import cv2

# 1️⃣ 创建 MuJoCo 物理环境
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

# 2️⃣ 机器人 & 猫猫初始速度
robot_speed = 0.02
cats = [
    {"id": 1, "vx": 0.03, "vy": 0.02},
    {"id": 2, "vx": -0.02, "vy": 0.04}
]

# 3️⃣ 创建 OpenCV 视频录制
fps = 30
video_writer = cv2.VideoWriter("turing_cat_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

# 4️⃣ 模拟循环 & 录制视频
for step in range(300):  # 运行 10 秒（30fps）
    # 机器人朝目标移动
    data.qpos[0] += robot_speed  # x 轴前进

    # 猫猫运动
    for cat in cats:
        cat_id = cat["id"]
        data.qpos[cat_id * 2] += cat["vx"]  # x 方向移动
        data.qpos[cat_id * 2 + 1] += cat["vy"]  # y 方向移动

    # 计算 PGD 预测（简单预测 10 步后的猫猫位置）
    future_positions = []
    for cat in cats:
        fx, fy = data.qpos[cat["id"] * 2], data.qpos[cat["id"] * 2 + 1]
        for _ in range(10):
            fx += cat["vx"]
            fy += cat["vy"]
        future_positions.append((fx, fy))

    # 避障策略（如果预测猫猫在未来 10 步会撞到机器人，改变方向）
    for fx, fy in future_positions:
        if abs(fx - data.qpos[0]) < 0.1 and abs(fy - data.qpos[1]) < 0.1:
            robot_speed = -robot_speed  # 反方向移动，避免撞猫

    # 运行 MuJoCo 模拟
    mujoco.mj_step(model, data)

    # 录制帧
    img = mujoco.viewer.draw_frame(model, data)
    frame = np.flipud(np.array(img))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    video_writer.write(frame)

# 5️⃣ 关闭视频写入
video_writer.release()
print("视频已保存：turing_cat_test.mp4 🎥")
