import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATConv, GCNConv, SAGEConv

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_heads=4, dropout=0.3, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(TransformerConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        for _ in range(num_layers - 1):
            self.convs.append(TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * num_heads, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze()


class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_dim=32, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * heads, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze()


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze()


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.out(x).squeeze()
