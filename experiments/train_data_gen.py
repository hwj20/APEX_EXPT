import json
import numpy as np
import random


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


def generate_node(pos_range=10, speed_range=1.0):
    pos = np.random.uniform(-pos_range, pos_range, size=3)
    vel_dir = normalize(np.random.randn(3))
    speed = np.random.uniform(0.5, speed_range)
    return pos, vel_dir * speed


def compute_risk_score(pos_master, pos_tgt, vel_tgt, dis_w=0.34, dir_w=0.33, vel_w=0.33):
    dist_vec = pos_tgt - pos_master
    dist = np.linalg.norm(dist_vec)

    # === 距离分数 ===
    distance_score_raw = 1.0 / (dist + 1e-6)
    distance_score = np.clip(distance_score_raw, 0, 10) / 10.0  # 归一化到 0~1

    # === 朝向分数 ===
    direction_score_raw = np.dot(normalize(dist_vec), normalize(vel_tgt))  # ∈ [-1, 1]
    direction_score = (direction_score_raw + 1) / 2  # 映射到 0~1

    # === 速度大小分数 ===
    velocity_score_raw = np.linalg.norm(vel_tgt)  # ~ 0~1.5
    velocity_score = np.clip(velocity_score_raw / 1.5, 0, 1)  # 归一化到 0~1

    # === 加权求和 ===
    return distance_score * dis_w + direction_score * dir_w + velocity_score * vel_w



cnt = 0
tt = 0


def create_reverse_sample(collision=True, dt=3.0, num_nodes=6, threshold =0.7):
    master_idx = 0
    pos_t_list, vel_list = [], []
    for i in range(num_nodes):
        pos, vel = generate_node()
        pos_t_list.append(pos)
        vel_list.append(vel)

    pos_m = pos_t_list[master_idx]
    vel_m = vel_list[master_idx]
    pos_m_future = pos_m + vel_m * dt
    # 强制制造碰撞场景
    if collision:
        tgt_idx = random.randint(1, num_nodes - 1)
        pos_t = pos_t_list[tgt_idx]
        dir_to_future_master = normalize(pos_m_future - pos_t)
        # 改成 dis/dt
        dist_to_master = np.linalg.norm(pos_m_future - pos_t)
        speed = dist_to_master / dt

        vel_list[tgt_idx] = dir_to_future_master * speed
        pos_tgt_dt = pos_t + speed * dir_to_future_master * dt
        score = compute_risk_score(pos_m_future, pos_tgt_dt, speed*dir_to_future_master)
        label = 1 if score > threshold else 0  # danger threshold 可调
        # global cnt, tt
        # if label == 1:
        #     cnt += 1
        #     if collision:
        #         tt += 1
        #     print(cnt, score, collision, tt)

    pos_t_dt_list = [pos + vel * dt for pos, vel in zip(pos_t_list, vel_list)]
    edge_index = [[master_idx] * (num_nodes - 1), [i for i in range(num_nodes) if i != master_idx]]

    edge_label = []
    for tgt_idx in edge_index[1]:
        pos_m = pos_t_list[master_idx]
        pos_tgt = pos_t_list[tgt_idx]
        pos_tgt_dt = pos_t_dt_list[tgt_idx]
        vel_tgt = pos_tgt_dt - pos_tgt
        score = compute_risk_score(pos_m_future, pos_tgt_dt, vel_tgt)
        label = 1 if score > threshold else 0  # danger threshold 可调
        edge_label.append(label)
        # global cnt, tt
        # if label == 1:
        #     cnt += 1
        #     if collision:
        #         tt += 1
        # print(cnt, score, collision, tt)

    node_features_t = [[1.0 if i == master_idx else 0.0] + pos.tolist() + vel.tolist()
                       for i, (pos, vel) in enumerate(zip(pos_t_list, vel_list))]
    node_features_t_dt = [[1.0 if i == master_idx else 0.0] + pos.tolist() + vel.tolist()
                          for i, (pos, vel) in enumerate(zip(pos_t_dt_list, vel_list))]

    edge_attr_t = [(pos_t_list[j] - pos_t_list[master_idx]).tolist() for j in edge_index[1]]
    edge_attr_t_dt = [(pos_t_dt_list[j] - pos_t_list[master_idx]).tolist() for j in edge_index[1]]

    return {
        "node_features_t": node_features_t,
        "node_features_t_dt": node_features_t_dt,
        "edge_index": edge_index,
        "edge_attr_t": edge_attr_t,
        "edge_attr_t_dt": edge_attr_t_dt,
        "edge_label": edge_label
    }


def generate_dataset(n_samples=500, seed=42, save_path="reverse_graphormer_data.json"):
    random.seed(seed)
    np.random.seed(seed)
    dataset = []

    for i in range(n_samples):
        sample = create_reverse_sample(collision=(i % 2 == 0))  # 一半 collision，一半 safe
        dataset.append(sample)

    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Saved {len(dataset)} samples to {save_path}")


if __name__ == "__main__":
    generate_dataset(n_samples=500)
