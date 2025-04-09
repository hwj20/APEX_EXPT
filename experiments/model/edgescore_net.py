import torch
from torch import nn
import torch.nn.functional as F

class EdgeScoreNet(nn.Module):
    def __init__(self, in_feats=7, edge_feat_dim=3, hidden_dim=32, num_heads=4, dropout=0.3):
        super(EdgeScoreNet, self).__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t, x_t_dt, edge_index, dt=1.0):
        """
        x_t: [num_nodes, 7]
        x_t_dt: [num_nodes, 7]
        edge_index: [2, num_edges]
        dt: timestep
        """
        master_mask = x_t[:, 0] == 1
        assert master_mask.sum() == 1, "必须有且仅有一个主节点"
        master_idx = master_mask.nonzero(as_tuple=False).squeeze()

        pos_t = x_t[:, 1:4]
        pos_t_dt = x_t_dt[:, 1:4]
        vel_t = F.normalize(pos_t_dt - pos_t, dim=-1) / dt

        master_pos = pos_t[master_idx].unsqueeze(0)
        tgt_idx = edge_index[1]
        tgt_pos = pos_t[tgt_idx]
        tgt_vel = vel_t[tgt_idx]

        rel_vec = tgt_pos - master_pos
        dist = torch.norm(rel_vec, dim=-1, keepdim=True) + 1e-6
        direction_score = F.cosine_similarity(rel_vec, tgt_vel, dim=-1, eps=1e-6).unsqueeze(1)
        velocity_score = torch.norm(pos_t_dt[tgt_idx] - pos_t[tgt_idx], dim=-1, keepdim=True)
        distance_score = 1.0 / dist

        edge_attr = torch.cat([distance_score, direction_score, velocity_score], dim=-1)
        edge_pred = torch.sigmoid(self.edge_mlp(edge_attr)).view(-1)
        return edge_pred
