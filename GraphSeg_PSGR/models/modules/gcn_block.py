import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv


class GCN(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dropout_ratio=0.5):
        super(GCN, self).__init__()
        self.dropout = dropout_ratio
        if inter_channels is None:
            inter_channels = in_channels

        self.gc1 = GCNConv(in_channels, inter_channels)
        self.gc2 = GCNConv(inter_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.gc1.forward(x=x, edge_index=edge_index, edge_weight=edge_weight))
        # x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc2.forward(x=x, edge_index=edge_index, edge_weight=edge_weight))
        # x = self.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class HOGCN(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dropout_ratio=0.5):
        super(HOGCN, self).__init__()
        self.dropout = dropout_ratio
        if inter_channels is None:
            inter_channels = in_channels

        self.gc1 = GraphConv(in_channels, inter_channels)
        self.gc2 = GraphConv(inter_channels, in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.gc1.forward(x=x, edge_index=edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.gc2.forward(x=x, edge_index=edge_index, edge_weight=edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
