import pybullet as p
import pybullet_data
import time
import networkx as nx

# 初始化 PyBullet 物理引擎
def init_simulation():
    p.connect(p.GUI)  # GUI 模式方便观察
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # 加载场景
    plane = p.loadURDF("plane.urdf")
    table1 = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
    table2 = p.loadURDF("table/table.urdf", basePosition=[2, 0, 0])  # 目标桌子

    # 放置多个水杯
    cup1 = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0, 0, 1])  # 稳定的杯子
    cup2 = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[-0.3, 0, 1])  # 可能掉的杯子
    cup3 = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.3, 0, 1])  # 另一个稳定杯子

    # 让 cup2 直接悬空，等它掉落
    cup_falling = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[-0.5, 0, 1.2])

    robot = p.loadURDF("r2d2.urdf", basePosition=[-1, 0, 0])

    return table1, table2, [cup1, cup2, cup3, cup_falling], robot

# 运行物理模拟，计算杯子多久掉落
def run_simulation(cups, steps=300):
    fall_times = {}
    for cup in cups:
        fall_times[cup] = None

    for step in range(steps):
        p.stepSimulation()
        time.sleep(1 / 240)
        for cup in cups:
            pos, _ = p.getBasePositionAndOrientation(cup)
            if pos[2] < 0.5 and fall_times[cup] is None:
                fall_times[cup] = step * (1 / 240)  # 记录掉落时间

    return fall_times

# 机器人任务规划
def apply_grounded_decoding(task_plan, fall_times):
    if any(fall_times.values()):
        return "机器人调整任务：先抢救快掉落的杯子，再执行任务。"
    return task_plan

# 主逻辑
table1, table2, cups, robot = init_simulation()
fall_times = run_simulation(cups)
print("杯子掉落时间：", fall_times)

physics_graph = nx.Graph()
physics_graph.add_edge("robot", "cup", weight=0.5)
physics_graph.add_edge("table1", "cup", weight=1.0)
physics_graph.add_edge("table2", "cup", weight=0.8)

adjusted_plan = apply_grounded_decoding("搬运所有杯子到目标桌", fall_times)

print("最终任务计划：", adjusted_plan)

# 断开 PyBullet 连接
p.disconnect()
