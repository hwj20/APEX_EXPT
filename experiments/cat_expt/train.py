import torch
from torch_geometric.data import Data, DataLoader
from experiments.cat_expt.model.graphormer import DiffGraphormer, FocalLoss  # 请确保路径正确
import json

# === 参数配置 ===
BATCH_SIZE = 1
HIDDEN_DIM = 32
NUM_HEADS = 4
EDGE_FEAT_DIM = 3
NODE_FEAT_DIM = 7
EPOCHS = 100
LR = 1e-3
SEED = 2025
MODEL_SAVE_PATH = "model/diffgraphormer_physics.pt"
DATA_PATH = "model/data/reverse_graphormer_data.json"

# === 设置设备 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)


# === 加载数据 ===
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
        edge_label = torch.tensor(sample['edge_label'], dtype=torch.float)

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
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        # logits = model(data.edge_attr_t)
        logits = model(data.x_t, data.x_t_dt, data.edge_index, dt)
        loss = criterion(logits, data.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


threshold = 0.5


# === 验证函数 ===
@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    total = correct = 0
    true_positives = 0
    false_negatives = 0
    positives = 0

    for data in loader:
        data = data.to(device)
        logits = model(data.x_t, data.x_t_dt, data.edge_index, dt)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

        correct += (pred == data.edge_label).sum().item()
        total += len(pred)

        # Recall 部分
        true_positives += ((pred == 1) & (data.edge_label == 1)).sum().item()
        false_negatives += ((pred == 0) & (data.edge_label == 1)).sum().item()
        positives += (data.edge_label == 1).sum().item()

    acc = correct / total if total > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return acc, recall



# === 主程序 ===
if __name__ == "__main__":
    dt = 0.01
    torch.manual_seed(SEED)
    dataset = load_dataset(DATA_PATH)
    print(
        f"⚖️ Positive ratio: {sum(d.edge_label.sum().item() for d in dataset) / sum(len(d.edge_label) for d in dataset):.3f}")

    split = int(0.8 * len(dataset))
    train_loader = DataLoader(dataset[:split], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[split:], batch_size=BATCH_SIZE)

    model = DiffGraphormer(
        in_feats=NODE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        edge_feat_dim=EDGE_FEAT_DIM
    ).to(device)

    pos_ratio = sum(d.edge_label.sum().item() for d in dataset) / sum(len(d.edge_label) for d in dataset)
    pos_weight = torch.tensor([1.0 / pos_ratio - 1], device=device)  # pos_ratio≈0.44 → pos_weight≈1.27
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterion = FocalLoss(alpha=pos_ratio, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🔥 Training Start!")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc,recall = evaluate(model, val_loader)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f} | Recall: {recall:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
