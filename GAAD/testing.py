import torch
from torch_geometric.data import Data
from GraphLayer import *

# Define the number of nodes and features
node_number = 12
node_features = 5

# Generate random node features
x = torch.randn(node_number, node_features)

# Generate random edge indices
edge_index = torch.randint(node_number, (2, node_number * 2))

graph_layer = GraphLayer(node_features, 1)
# Perform forward pass
output = graph_layer(x, edge_index)
print(output.shape)
