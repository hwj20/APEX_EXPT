import numpy as np
import torch
import torch.nn.functional as F
from .mujoco_perception import simulator


class APEX:
    def __init__(self, graphormer_model, physics_simulator, llm_agent, dt=0.01, device="cpu"):
        self.graphormer = graphormer_model.to(device)
        self.physics_sim = simulator(physics_simulator)
        self.llm_agent = llm_agent
        self.device = device
        self.last_trigger = None
        self.dt = dt

    def construct_graph(self, snapshot, snapshot_dt, dt=0.1):
        """
        输入:
        - snapshot: 当前环境状态 {'objects': [...]}
        - snapshot_dt: 下一帧环境状态 {'objects': [...]}
        - dt: 时间间隔
        返回:
        - x_t: 当前帧特征
        - x_t_dt: 下一帧特征
        - edge_index: 边信息 [from, to]
        """

        node_features_t = []
        node_features_t_dt = []
        edge_index = [[], []]

        objs = snapshot["objects"]
        objs_dt = snapshot_dt["objects"]

        master_name = "robot"
        master_idx = None

        for idx, obj in enumerate(objs):
            name = obj["name"]
            pos = np.array(obj["position"][:3])
            pos_dt = np.array(objs_dt[idx]["position"][:3])
            vel_recompute = (pos_dt - pos) / dt

            is_master = 1.0 if name == master_name else 0.0
            if is_master:
                master_idx = idx

            node_feat_t = [is_master] + pos.tolist() + vel_recompute.tolist()
            node_feat_t_dt = [is_master] + pos_dt.tolist() + vel_recompute.tolist()

            node_features_t.append(node_feat_t)
            node_features_t_dt.append(node_feat_t_dt)

        # 构造 master -> other 的边
        for idx in range(len(objs)):
            if idx == master_idx:
                continue
            edge_index[0].append(master_idx)
            edge_index[1].append(idx)

        x_t = torch.tensor(node_features_t, dtype=torch.float32).to(self.device)
        x_t_dt = torch.tensor(node_features_t_dt, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)

        return x_t, x_t_dt, edge_index

    def compute_attention(self, x_t, x_t_dt, edge_index, dt=1.0):
        """
        Run the graphormer model to obtain edge danger scores as attention proxy.
        """
        with torch.no_grad():
            scores = self.graphormer(x_t, x_t_dt, edge_index, dt)
        return scores.cpu().numpy()  # convert to list for processing

    def select_focused_graph(self, edge_index, attention_scores, k, threshold=0.7):
        """
        Select top-k edges based on attention scores.
        """
        edge_tuples = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        sorted_edges = sorted(zip(edge_tuples, attention_scores), key=lambda x: -x[1])
        top_edges = sorted_edges[:k]
        if max(attention_scores) < threshold:
            return None
        else:
            return {
                "nodes": list(set([i for edge, _ in top_edges for i in edge])),
                "edges": [edge for edge, _ in top_edges]
            }

    def generate_physical_summary(self, focused_graph):
        """
        Convert focused subgraph to a textual summary.
        """
        edge_descriptions = [f"Object {src} may collide with Object {tgt}" for src, tgt in focused_graph["edges"]]
        return "Potential interactions:\n" + "\n".join(edge_descriptions)

    def enumerate_actions(self, state):
        """
        枚举所有可执行动作
        返回：
        {
            "动作名称": {
                "velocity": [vx, vy, vz],
                "duration": time (s),
                "description": "文字描述"
            }
        }
        """

        return {
            "move_left": {
                "velocity": [-3.0, 0.0, 0.0],
                "duration": 1.0,
                "description": "move left with velocity[x] = -1.0 for 1s"
            },
            "move_right": {
                "velocity": [3.0, 0.0, 0.0],
                "duration": 1.0,
                "description": "move right with velocity[x] = 1.0 for 1s"
            },
            "move_up": {
                "velocity": [0.0, 3.0, 0.0],
                "duration": 1.0,
                "description": "move up with velocity[y] = 1.0 for 1s"
            },
            "move_down": {
                "velocity": [0.0, -3.0, 0.0],
                "duration": 1.0,
                "description": "move down with velocity[y] = -1.0 for 1s"
            },
            "jump": {
                "velocity": [0.0, 0.0, 3.0],
                "duration": 0.2,
                "description": "jump with velocity[z] = 1.0 for 0.2s, then fall"
            },
            "stay": {
                "velocity": [0.0, 0.0, 0.0],
                "duration": 1.0,
                "description": "stay still for 1s"
            }
        }

    def simulate_action(self, model, env_data, action):
        return self.physics_sim.sim(model, env_data, action)

    def describe_simulation(self, result: dict) -> str:
        """
        输入：
            result: mujoco_sim返回的结果字典
        输出：
            人类友好的字符串，总结每个动作的效果
        """

        summary = []

        for action, info in result.items():
            robot_pos = None
            cat_pos_list = []

            for obj in info["final_robot_pos"]:
                if obj["name"] == "robot":
                    robot_pos = np.array(obj["position"])
                elif "cat" in obj["name"]:
                    cat_pos_list.append(np.array(obj["position"]))

            min_dist = min(np.linalg.norm(robot_pos[:2] - cat[:2]) for cat in cat_pos_list)
            height = robot_pos[2]

            safe_str = "Safe" if min_dist > 0.5 else "Danger"
            jump_str = "Jumped" if height > 0.2 else "Stayed ground"

            summary.append(f"- Action [{action}]: "
                           f"Distance to nearest cat = {min_dist:.2f}m, "
                           f"Height = {height:.2f}m, "
                           f"Assessment = {safe_str}, {jump_str}")

        return "\n".join(summary)

    def decode_move(self, decision: str):
        """
        输入 LLM 决策的字符串，返回 {'velocity': [...], 'duration': ...}
        """
        decision = decision.lower()  # 防止大小写影响

        # stay
        vel = [0.0, 0.0, 0.0]  # x, y, z
        duration = 1.0  # s

        if "left" in decision:
            vel[0] = -3.0
        elif "right" in decision:
            vel[0] = 3.0
        elif "up" in decision:
            vel[1] = 3.0
        elif "down" in decision:
            vel[1] = -3.0
        elif "jump" in decision:
            vel[2] = 3.0  # jump
            duration = 0.2

        return {"velocity": vel, "duration": duration}

    def run(self, snapshot_t, snapshot_t_dt, dt, physical_model, env_data):
        x_t, x_t_dt, edge_index = self.construct_graph(snapshot_t, snapshot_t_dt, dt)

        attention_scores = self.compute_attention(x_t, x_t_dt, edge_index)

        focused_graph = self.select_focused_graph(edge_index, attention_scores, k=5)

        # not trigger
        if focused_graph is None:
            return False, "stay"

        summary = self.generate_physical_summary(focused_graph)
        actions = self.enumerate_actions(snapshot_t)

        sim_result = self.simulate_action(physical_model, env_data, actions)
        results = self.describe_simulation(sim_result)

        decision = self.llm_agent.decide_move_apex(snapshot_t, summary, actions, results)
        move = self.decode_move(decision)

        return True, move
