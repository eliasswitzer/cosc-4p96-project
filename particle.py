import numpy as np

class Particle:
  def __init__(self, search_bounds,init = None):
    self.search_bounds = search_bounds
    num_dims = len(search_bounds)

    if init is None:
      self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds]) # initialize position randomly within bounds
    else:
      self.position = np.array(list(init.values()))
    self.velocity = np.zeros(len(search_bounds)) # initialize velocities to zero

    self.best_position = self.position.copy()
    self.best_fitness = float('inf')

  def get_network_params(self):
    return {
        'num_hidden_layers': int(np.round(self.position[0])),
        'hidden_layer_size': int(np.round(self.position[1])),
        'learning_rate': self.position[2],
        'momentum': self.position[3],
        'batch_size': int(np.round(self.position[4])),
        'weight_decay': self.position[5],
        'dropout_rate': self.position[6]
    }