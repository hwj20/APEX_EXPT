import matplotlib.pyplot as plt
import networkx as nx
import os

import numpy as np
import torch
from .mujoco_perception import simulator


class APEX:
    def __init__(self, graphormer_model, physics_simulator, llm_agent, available_move, dt=0.01, device="cpu"):
        self.graphormer = graphormer_model.to(device)
        self.physics_sim = simulator(physics_simulator)
        self.llm_agent = llm_agent
        self.device = device
        self.last_trigger = None
        self.dt = dt
        self.available_move = available_move

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
            vel_dir = vel_recompute / (np.linalg.norm(vel_recompute) + 1e-6)

            is_master = 1.0 if name == master_name else 0.0
            if is_master:
                master_idx = idx

            node_feat_t = [is_master] + pos.tolist() + vel_dir.tolist()
            node_feat_t_dt = [is_master] + pos_dt.tolist() + vel_dir.tolist()

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

    def visualize_attention(self, x_t, edge_index, scores, step):
        G = nx.DiGraph()
        node_pos = {i: x_t[i, 1:3].cpu().numpy() for i in range(x_t.shape[0])}

        for i in range(x_t.shape[0]):
            G.add_node(i)

        edge_list = edge_index.t().cpu().numpy().tolist()
        for (src, tgt), w in zip(edge_list, scores):
            G.add_edge(src, tgt, weight=w)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        pos = node_pos

        plt.figure(figsize=(5, 5))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                edge_color=edge_weights, edge_cmap=plt.cm.Reds, width=2)
        plt.title("Graphormer Attention Heatmap")
        os.makedirs("visualization", exist_ok=True)
        plt.savefig(f"visualization/attention_step_{step}.png")
        plt.close()
        print(f"attention saved into visualization/attention_step_{step}.png")

    def compute_attention(self, x_t, x_t_dt, edge_index, dt=1.0, save_visual=False, step=0):
        """
        Run the graphormer model to obtain edge danger scores as attention proxy.
        """
        with torch.no_grad():
            scores = self.graphormer(x_t, x_t_dt, edge_index, dt)

        if save_visual:
            self.visualize_attention(x_t, edge_index, scores.cpu().numpy(), step)

        return scores.cpu().numpy()  # convert to list for processing

    def select_focused_graph(self, edge_index, attention_scores, k, threshold=0.5):
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
        edge_descriptions = []
        for src, tgt in focused_graph["edges"]:
            if src == 0:
                edge_descriptions = f"Robot {src} may collide with cat {tgt}"
        return "Potential interactions:\n" + "\n".join(edge_descriptions)

    def simulate_action(self, model, env_data, action):
        return self.physics_sim.sim(model, env_data, action)

    def describe_simulation(self, result: dict) -> str:
        """
        describe the simulation results into natural language that LLMs can understand
        """

        summary = []

        for action, info in result.items():
            robot_pos = None
            cat_pos_list = []

            for obj in info["final_pos"]:
                if obj["name"] == "robot":
                    robot_pos = np.array(obj["position"])
                elif "cat" in obj["name"]:
                    cat_pos_list.append(np.array(obj["position"]))

            min_dist = min(np.linalg.norm(robot_pos[:3] - cat[:3]) for cat in cat_pos_list)
            height = robot_pos[2]

            safe_str = "Safe" if min_dist > 0.4 else "Danger"
            jump_str = "Jumped" if height > 0.3 else "Ground"

            if info['description']['duration'] <= 0:
                summary.append(f"- Action [{action}]: "
                               f"Assessment = Invalid")
            else:
                summary.append(f"- Action [{action}]: "
                               f"Max Duration:{info['description']['duration']}, "
                               f"Distance to nearest cat = {min_dist:.2f}m, "
                               f"Height = {height:.2f}m, "
                               f"Assessment = {safe_str}, {jump_str}")

        return "\n".join(summary)

    def run(self, snapshot_t, snapshot_t_dt, dt, physical_model, env_data, step):
        x_t, x_t_dt, edge_index = self.construct_graph(snapshot_t, snapshot_t_dt, dt)

        save_visual = step % 50 == 0
        # save_visual = False
        attention_scores = self.compute_attention(x_t, x_t_dt, edge_index, dt, save_visual=save_visual, step=step)

        focused_graph = self.select_focused_graph(edge_index, attention_scores, k=5)

        # not trigger
        if focused_graph is None:
            return False, "stay", True

        summary = self.generate_physical_summary(focused_graph)
        actions = self.available_move

        sim_result = self.simulate_action(physical_model, env_data, actions)
        results = self.describe_simulation(sim_result)
        print(results)

        move, valid_move = self.llm_agent.decide_move_apex(snapshot_t, summary, actions, results)

        self.last_trigger = step

        return True, move, valid_move
