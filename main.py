import networkx as nx
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt

# 初始化 PyBullet 物理环境
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 创建物理环境（多米诺骨牌）
plane_id = p.loadURDF("plane.urdf")
domino_ids = []
start_pos = [-1, 0, 0.1]
domino_gap = 0.15

for i in range(5):  # 生成 5 个多米诺骨牌
    domino_id = p.loadURDF("cube_small.urdf",
                           [start_pos[0] + i * domino_gap, start_pos[1], start_pos[2]],
                           p.getQuaternionFromEuler([0, 0, 0]))
    domino_ids.append(domino_id)

# 创建物理交互图
G = nx.DiGraph()

for i in range(len(domino_ids) - 1):
    G.add_edge(f"Domino {i+1}", f"Domino {i+2}", weight=9.8)  # 代表重力作用

# 可视化交互图
plt.figure(figsize=(5, 3))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Physics Interaction Graph (Domino Effect)")
plt.show()

# 运行物理仿真
for _ in range(100):  # 让多米诺倒下
    p.applyExternalForce(domino_ids[0], -1, [5, 0, 0], [0, 0, 0], p.WORLD_FRAME)
    p.stepSimulation()
    time.sleep(1/240)  # 物理时间步

# 断开连接
p.disconnect()

# 生成 LLM 预测的文本输出
pgd_prediction = "The first domino falls, transferring energy to the next, eventually causing all dominos to fall in sequence."

# 输出 LLM 物理推理结果
pgd_prediction
