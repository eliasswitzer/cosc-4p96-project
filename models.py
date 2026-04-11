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

#DEFINE NN CLASS
class ParticlePredictor(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(7,40),
        nn.LeakyReLU(0.1),
        nn.Linear(40,10),
        nn.LeakyReLU(0.1),
        nn.Linear(10,1),

    )
  def forward(self,x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

