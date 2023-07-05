import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from dataset import HW3Dataset


dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]

# Model parameters
channels = 12
num_heads = 15
dropout = 0.1

# Normalize node year and add it the features
years = data.node_year
min_year = torch.min(years)
max_year = torch.max(years)
normalized_years = (years - min_year) / (max_year - min_year)
normalized_years = normalized_years.to(data.x.dtype)

# Concatenate
features = torch.cat((data.x, normalized_years), dim=1)


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.norm = torch.nn.BatchNorm1d(dataset.num_features + 1)
        self.conv1 = GATConv(dataset.num_features + 1, hidden_channels, heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads)

    def forward(self, x, edge_index):
        x = self.norm(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# Load model
model = GAT(hidden_channels=channels, heads=num_heads)
model.load_state_dict(torch.load('./model.pkl'))

out = model(features, data.edge_index)
out = F.log_softmax(out, dim=1)
preds = out.argmax(dim=1)

# correct = preds == data.y.squeeze()
# accuracy = int(correct.sum()) / len(correct)
# print("Accuracy:", accuracy)

# Export prediction results
prediction_dict = {'idx': np.arange(len(data.x)).tolist(), 'prediction': preds.tolist()}
prediction_df = pd.DataFrame(prediction_dict)
prediction_df.to_csv('./prediction.csv', index=False, header=True)
