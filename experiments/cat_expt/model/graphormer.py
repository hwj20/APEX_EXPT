import torch
from torch import nn
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F

class DiffGraphormer(nn.Module):
    def __init__(self, in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3):
        super(DiffGraphormer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.node_encoder = nn.Linear(in_feats, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.edge_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_t, x_t_dt, edge_index, dt):
        """
        x_t: [num_nodes, 7] 包含是否主节点 (1维), pos_t (3维), velocity_unit (3维)
        x_t_dt: [num_nodes, 7] 其实我们只用 pos_t_dt 来计算速度大小
        edge_index: [2, num_edges] (from, to)
        """
        master_mask = x_t[:, 0] == 1
        assert master_mask.sum() == 1, "必须有且仅有一个主节点"
        master_idx = master_mask.nonzero(as_tuple=False).squeeze()

        pos_t = x_t[:, 1:4]         # 当前帧位置
        pos_t_dt = x_t_dt[:, 1:4]   # 下一帧位置
        vel_t = F.normalize(pos_t_dt - pos_t, dim=-1) /dt # 单位速度方向向量

        master_pos = pos_t[master_idx].unsqueeze(0)         # shape: [1, 3]
        master_vel = vel_t[master_idx].unsqueeze(0)         # shape: [1, 3]

        # Edge 计算（每条边是从 master_idx 指向某节点）
        tgt_idx = edge_index[1]  # [num_edges]
        tgt_pos = pos_t[tgt_idx]
        tgt_vel = vel_t[tgt_idx]

        rel_vec = master_pos-tgt_pos  # [num_edges, 3]
        dist = torch.norm(rel_vec, dim=-1, keepdim=True) + 1e-6
        direction_score = F.cosine_similarity(rel_vec, tgt_vel, dim=-1, eps=1e-6).unsqueeze(1)  # [num_edges, 1]
        velocity_score = torch.norm(pos_t_dt[tgt_idx] - pos_t[tgt_idx], dim=-1, keepdim=True)
        distance_score = 1.0 / dist  # [num_edges, 1]

        edge_attr = torch.cat([distance_score, direction_score, velocity_score], dim=-1)  # [num_edges, 3]
        edge_feat = self.edge_encoder(edge_attr).view(-1, self.num_heads, self.hidden_dim // self.num_heads)

        x_embed = self.node_encoder(x_t)
        x_trans = self.transformer(x_embed, edge_index, edge_feat)
        x_trans = self.dropout(x_trans)

        edge_repr = x_trans[edge_index[0]] + x_trans[edge_index[1]]  # [num_edges, hidden_dim]
        edge_pred = torch.sigmoid(self.edge_classifier(edge_repr))   # [num_edges, 1]
        return edge_pred.view(-1)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 正类权重
        self.gamma = gamma  # focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (before sigmoid), shape [batch]
        targets: binary labels, shape [batch]
        """
        probs = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    num_nodes = 6
    num_edges = 5
    node_feat_dim = 7
    edge_feat_dim = 3
    dt = 0.01

    x_t = torch.randn((num_nodes, node_feat_dim))
    x_t_dt = torch.randn((num_nodes, node_feat_dim))
    x_t[0, 0] = 1.0  # 设第 0 个是主节点，其余为 0
    x_t[1:, 0] = 0.0

    edge_index = torch.tensor([[0]*num_edges, [1, 2, 3, 4, 5]], dtype=torch.long)

    model = DiffGraphormer()
    preds = model(x_t, x_t_dt, edge_index,dt)
    print("⚡ Predicted danger scores:", preds)
