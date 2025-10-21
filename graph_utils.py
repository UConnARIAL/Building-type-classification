import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

def build_knn_graph(X, coords, y, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    edges = [(i, j) for i, row in enumerate(indices) for j in row[1:]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y.astype(np.float32), dtype=torch.float)
    )
