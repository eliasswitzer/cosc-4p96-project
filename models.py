import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim, num_classes, num_layers, hidden_size, dropout_rate):
    super().__init__()
    layers = []
    next_num_features = input_dim

    for i in range(num_layers):
      layers.append(nn.Linear(next_num_features, hidden_size))
      layers.append(nn.BatchNorm1d(hidden_size))
      layers.append(nn.ReLU())
      layers.append(nn.Dropout(dropout_rate))
      next_num_features = hidden_size

    layers.append(nn.Linear(hidden_size, num_classes))
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x.view(x.size(0), -1)) # flatten and pass through network