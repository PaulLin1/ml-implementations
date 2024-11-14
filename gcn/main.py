import torch
from torch_geometric.datasets import CoraFull
from torch_geometric.data import DataLoader

dataset = CoraFull(root="cora")
data = dataset[0]
loader = DataLoader(dataset, batch_size=31, shuffle=True)

class GraphSageLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphSageLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        # Sums neighbor features and aggregates
        # neighbor_weighted_averages = torch.mean()
        neighbor_messages = torch.mean(dsfsd)
        message = torch.concat((x, neighbor_messages), 0)
        weighted_message = torch.matmul(self.w, message) 
        return torch.nn.functional.relu(weighted_message)

layer = GraphSageLayer(8, 4)
z = torch.ones(8)
print(layer(z))
