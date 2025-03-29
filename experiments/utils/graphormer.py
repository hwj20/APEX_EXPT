import torch
from torch_geometric.nn import TransformerConv
import torch.nn as nn

class DiffGraphormer(nn.Module):
    def __init__(self, in_feats, hidden_dim, num_classes, num_heads, edge_feat_dim, dropout=0.3):
        super(DiffGraphormer, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.node_encoder = nn.Linear(in_feats, hidden_dim)
        self.edge_diff_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.transformer = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.edge_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_t, x_t_dt, edge_index, edge_attr_t, edge_attr_t_dt):
        x = self.node_encoder(x_t)
        edge_diff = edge_attr_t_dt - edge_attr_t
        edge_encoded = self.edge_diff_encoder(edge_diff)
        edge_encoded = edge_encoded.view(-1, self.num_heads, self.hidden_dim // self.num_heads)
        x_trans = self.transformer(x, edge_index, edge_encoded)
        x_trans = self.dropout(x_trans)
        edge_repr = x_trans[edge_index[0]] + x_trans[edge_index[1]]
        edge_pred = self.edge_classifier(edge_repr)
        return edge_pred

if __name__ == "__main__":
    torch.manual_seed(42)
    num_nodes = 4
    num_edges = 4
    in_feats = 3
    edge_feat_dim = 2
    hidden_dim = 8
    num_heads = 2
    num_classes = 1

    x_t = torch.randn((num_nodes, in_feats))
    x_t_dt = torch.randn((num_nodes, in_feats))
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_attr_t = torch.randn((num_edges, edge_feat_dim))
    edge_attr_t_dt = torch.randn((num_edges, edge_feat_dim))

    model = DiffGraphormer(in_feats, hidden_dim, num_classes, num_heads, edge_feat_dim)
    out = model(x_t, x_t_dt, edge_index, edge_attr_t, edge_attr_t_dt)
    print(out)
