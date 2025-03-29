import json
import random
import numpy as np
from typing import List, Dict
import os


def generate_graph_data(num_nodes: int, num_edges: int) -> Dict:
    """
    Generate a synthetic graph with node features and edge attributes.
    """
    nodes = [np.random.randn(4).tolist() for _ in range(num_nodes)]  # 4-dim node features
    edges = []
    edge_attrs = []

    connected = set()
    while len(edges) < num_edges:
        i, j = random.sample(range(num_nodes), 2)
        if (i, j) not in connected and (j, i) not in connected:
            edges.append([i, j])
            edge_attrs.append(np.random.randn(2).tolist())  # 2-dim edge attributes
            connected.add((i, j))

    return {
        "nodes": nodes,
        "edges": edges,
        "edge_attrs": edge_attrs
    }


def generate_diff_labels(edge_attrs_1: List[List[float]], edge_attrs_2: List[List[float]]) -> List[int]:
    """
    Compute binary labels for edge changes: 1 if L2 distance > threshold, else 0.
    """
    threshold = 0.5
    labels = []
    for a1, a2 in zip(edge_attrs_1, edge_attrs_2):
        dist = np.linalg.norm(np.array(a1) - np.array(a2))
        labels.append(int(dist > threshold))
    return labels


def create_dataset(n_samples: int, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    np.random.seed(seed)

    dataset = []
    for _ in range(n_samples):
        num_nodes = random.randint(4, 8)
        num_edges = random.randint(num_nodes, num_nodes * 2)

        G_t = generate_graph_data(num_nodes, num_edges)
        G_t_dt = generate_graph_data(num_nodes, num_edges)  # simulates small change
        edge_labels = generate_diff_labels(G_t["edge_attrs"], G_t_dt["edge_attrs"])

        dataset.append({
            "G_t": G_t,
            "G_t_dt": G_t_dt,
            "edge_labels": edge_labels
        })
    return dataset


# Save locally
dataset = create_dataset(10, seed=1234)
output_path = "graphormer_train_data_seed1234.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(output_path) # return file path for user
