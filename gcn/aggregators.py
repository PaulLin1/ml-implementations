import torch

class MeanAggregator(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MeanAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = torch.nn.Parameter(torch.empty(2 * in_features, out_features))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, input):
        x, sampled_neighbors = input
        neighbors_message = torch.mean(sampled_neighbors, dim=0)
        message = torch.concat((x, neighbors_message), 0)
        weighted_message = torch.matmul(message, self.w) 
        return torch.nn.functional.relu(weighted_message)
