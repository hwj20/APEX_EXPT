import json
import numpy as np
import random
from typing import List, Dict

def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec

def generate_node(pos_range=10, speed_range=1.0):
    pos = np.random.uniform(-pos_range, pos_range, size=3)
    vel_dir = normalize(np.random.randn(3))
    return pos, vel_dir

def compute_risk_score(pos_master, pos_tgt, vel_tgt, alpha=1.0, beta=1.0, gamma=1.0):
    dist_vec = pos_tgt - pos_master
    dist = np.linalg.norm(dist_vec)
    direction_score = np.dot(normalize(dist_vec), vel_tgt)
    distance_score = 1.0 / (dist + 1e-6)
    velocity_score = np.linalg.norm(vel_tgt)
    return alpha * distance_score + beta * direction_score + gamma * velocity_score

def create_sample(num_nodes=6, danger_threshold=0.5):
    master_idx = 0
    Δt = 1.0

    # === G_t ===
    pos_t_list = []
    vel_list = []
    node_features_t = []
    for i in range(num_nodes):
        pos, vel = generate_node()
        pos_t_list.append(pos)
        vel_list.append(vel)
        is_master = 1.0 if i == master_idx else 0.0
        node_feat = [is_master] + pos.tolist() + vel.tolist()
        node_features_t.append(node_feat)

    # === Edges: all from master to others ===
    edge_index = [[master_idx] * (num_nodes - 1), [i for i in range(num_nodes) if i != master_idx]]
    edge_index = np.array(edge_index)

    # === G_t+dt ===
    pos_t_dt_list = []
    node_features_t_dt = []
    edge_attr_t = []
    edge_attr_t_dt = []
    edge_label = []

    for i in range(num_nodes):
        pos_t_dt = pos_t_list[i] + vel_list[i] * Δt
        pos_t_dt_list.append(pos_t_dt)
        node_feat_dt = [1.0 if i == master_idx else 0.0] + pos_t_dt.tolist() + vel_list[i].tolist()
        node_features_t_dt.append(node_feat_dt)

    # === Edges & Labels ===
    for tgt_idx in edge_index[1]:
        pos_m, pos_tgt = pos_t_list[master_idx], pos_t_list[tgt_idx]
        pos_tgt_dt = pos_t_dt_list[tgt_idx]

        # original velocity
        vel_tgt = pos_tgt_dt - pos_tgt

        # current edge feature
        dist_vec_t = pos_tgt - pos_m
        edge_attr_t.append(dist_vec_t.tolist())

        # With prob 0.5 make it dangerous
        if random.random() > 0.5:
            # Move tgt closer toward master
            dir_to_master = normalize(pos_m - pos_tgt)
            pos_tgt_dt = pos_tgt + dir_to_master * np.random.uniform(1.5, 2.5)
            vel_tgt = pos_tgt_dt - pos_tgt

        dist_vec_dt = pos_tgt_dt - pos_m
        edge_attr_t_dt.append(dist_vec_dt.tolist())

        score = compute_risk_score(pos_m, pos_tgt_dt, normalize(vel_tgt))
        label = 1 if score > danger_threshold else 0
        edge_label.append(label)
        # print(label)

    return {
        "node_features_t": node_features_t,
        "node_features_t_dt": node_features_t_dt,
        "edge_index": edge_index.tolist(),
        "edge_attr_t": edge_attr_t,
        "edge_attr_t_dt": edge_attr_t_dt,
        "edge_label": edge_label
    }

def generate_dataset(n_samples=1000, seed=42, save_path="graphormer_physics_risk.json"):
    random.seed(seed)
    np.random.seed(seed)
    dataset = [create_sample() for _ in range(n_samples)]
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved {n_samples} samples to {save_path}")

if __name__ == "__main__":
    generate_dataset(n_samples=500, seed=2025)
