import torch
from torch_geometric.datasets import CoraFull
from torch_geometric.data import DataLoader
import aggregators

dataset = CoraFull(root="cora")
data = dataset[0]
loader = DataLoader(dataset, batch_size=32, shuffle=True)
layer = aggregators.MeanAggregator(len(data.x[0]), 4)
node = data.x[0]
node_neighbors = torch.stack([data.x[v] for u, v in zip(data.edge_index[0], data.edge_index[1]) if data.edge_index[0][u] == 0])
print(layer((node, node_neighbors)))
