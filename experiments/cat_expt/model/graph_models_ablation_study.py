import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATConv


class DiffGCN(nn.Module):
    """
    Simple GCN-based alternative for edge danger prediction.
    """
    def __init__(self, in_feats=7, hidden_dim=32, dropout=0.3):
        super(DiffGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(in_feats, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # edge classifier takes concatenated src+dst embeddings
        self.edge_classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_t, x_t_dt, edge_index, dt=None):
        # encode node features (ignoring dt dynamics for simplicity)
        x = self.node_encoder(x_t)
        x = F.relu(x)
        # graph convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # compute edge representations
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst]], dim=-1)  # [num_edges, hidden_dim*2]
        # classification
        scores = torch.sigmoid(self.edge_classifier(edge_repr)).view(-1)
        return scores


class DiffGAT(nn.Module):
    """
    GAT-based alternative for edge danger prediction.
    """
    def __init__(self, in_feats=7, hidden_dim=32, heads=4, dropout=0.3):
        super(DiffGAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.node_encoder = nn.Linear(in_feats, hidden_dim)
        # GATConv will produce hidden_dim output per head, we'll merge via concat
        self.gat = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # after concat, embedding dim = hidden_dim
        self.edge_classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_t, x_t_dt, edge_index, dt=None):
        # encode nodes
        x = self.node_encoder(x_t)
        x = F.elu(x)
        # graph attention
        x = self.gat(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        # edge repr
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst]], dim=-1)
        scores = torch.sigmoid(self.edge_classifier(edge_repr)).view(-1)
        return scores


# Quick test
if __name__ == "__main__":
    torch.manual_seed(0)
    num_nodes, num_edges = 6, 5
    x_t = torch.randn((num_nodes, 7))
    x_t[:, 0] = 0.0
    x_t[0, 0] = 1.0  # master node marker
    edge_index = torch.tensor([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5]], dtype=torch.long)

    for Model in [DiffGCN, DiffGAT]:
        model = Model()
        preds = model(x_t, x_t, edge_index, dt=0.01)
        print(f"{Model.__name__} predictions: {preds}")
