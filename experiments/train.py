import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from model.graphormer import DiffGraphormer  # 确保你把模型保存为 model.py
import json
from tqdm import tqdm
import os

# === 参数配置 ===
BATCH_SIZE = 1
HIDDEN_DIM = 64
NUM_HEADS = 4
EDGE_FEAT_DIM = 3
NODE_FEAT_DIM = 7
EPOCHS = 50
LR = 1e-4
SEED = 2025
MODEL_SAVE_PATH = "diffgraphormer_physics.pt"
DATA_PATH = "graphormer_physics_risk.json"

def focal_loss(pred, target, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Binary focal loss.
    pred: [batch_size] predicted probabilities (after sigmoid)
    target: [batch_size] ground truth (0 or 1)
    """
    pred = pred.clamp(min=eps, max=1. - eps)
    pt = pred * target + (1 - pred) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = -w * (1 - pt) ** gamma * pt.log()
    return loss.mean()

# === 设备选择 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# === 数据加载 ===
def load_dataset(path):
    with open(path, 'r') as f:
        raw = json.load(f)

    dataset = []
    for sample in raw:
        x_t = torch.tensor(sample['node_features_t'], dtype=torch.float)
        x_t_dt = torch.tensor(sample['node_features_t_dt'], dtype=torch.float)
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
        edge_attr_t = torch.tensor(sample['edge_attr_t'], dtype=torch.float)
        edge_attr_t_dt = torch.tensor(sample['edge_attr_t_dt'], dtype=torch.float)
        edge_label = torch.tensor(sample['edge_label'], dtype=torch.float)  # sigmoid -> float

        data = Data(
            x_t=x_t,
            x_t_dt=x_t_dt,
            edge_index=edge_index,
            edge_attr_t=edge_attr_t,
            edge_attr_t_dt=edge_attr_t_dt,
            edge_label=edge_label
        )
        dataset.append(data)
    return dataset

# === 训练函数 ===
def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x_t, data.x_t_dt, data.edge_index)
        # pred = model(data.x_t, data.x_t_dt, data.edge_index, data.edge_attr_t, data.edge_attr_t_dt)
        loss = F.binary_cross_entropy(pred,data.edge_label)
        # loss = focal_loss(pred, data.edge_label, alpha=0.25, gamma=2.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# === 验证函数 ===
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x_t, data.x_t_dt, data.edge_index)
        pred_label = (pred > 0.5).float()
        # print(pred_label)
        correct += (pred_label == data.edge_label).sum().item()
        total += len(pred)
    return correct / total if total > 0 else 0

# === 主程序入口 ===
if __name__ == "__main__":
    torch.manual_seed(SEED)

    dataset = load_dataset(DATA_PATH)
    split = int(0.8 * len(dataset))
    train_dataset = dataset[:split]
    val_dataset = dataset[split:]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DiffGraphormer(
        in_feats=NODE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        edge_feat_dim=EDGE_FEAT_DIM
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training Start!")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
