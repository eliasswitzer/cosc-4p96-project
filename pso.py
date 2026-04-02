import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# TODO: tidy up code and move some functions into the PSO class

class Particle:
  def __init__(self, search_bounds):
    self.search_bounds = search_bounds
    num_dims = len(search_bounds)

    self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds]) # initialize position randomly within bounds
    self.velocity = np.zeros(7) # initialize velocities to zero

    self.best_position = self.position
    self.best_fitness = float('inf')

  def get_network_params(self):
    return {
        'num_hidden_layers': int(self.position[0]),
        'hidden_layer_size': int(self.position[1]),
        'learning_rate': self.position[2],
        'momentum': self.position[3],
        'batch_size': int(self.position[4]),
        'weight_decay': self.position[5],
        'dropout_rate': self.position[6]
    }
  
class PSO:
  def __init__(self, num_particles, search_bounds, w=0.729, c1=1.49445, c2=1.49445):
    self.particles = [Particle(search_bounds=search_bounds) for _ in range(num_particles)]
    self.global_best_position = None
    self.global_best_fitness = float('inf')
    self.w, self.c1, self.c2, = w, c1, c2

  def optimize(self, fitness_function, num_iterations):

    # Evaluate initial population
    print(f"Evaluating Initial Population")
    for i in range(len(self.particles)):
      parameters = self.particles[i].get_network_params()
      fitness = fitness_function(parameters)
      print(f"Particle {i+1} | Validation Loss: {fitness:.4f} | Parameters: {parameters}")

      # Set Initial Personal Best
      self.particles[i].best_fitness = fitness
      self.particles[i].best_position = self.particles[i].position.copy()

      # Set Initial Global Best
      if fitness < self.global_best_fitness:
        self.global_best_position = self.particles[i].position.copy()
        self.global_best_fitness = fitness

    # Main PSO Loop
    for iteration in range(num_iterations): # TODO: add other stopping conditions (particle convergence)
      print(f"Iteration {iteration + 1}/{num_iterations}")

      for i in range(len(self.particles)):
        # Update Velocity
        r1 = np.random.rand(len(self.particles[i].position))
        r2 = np.random.rand(len(self.particles[i].position))

        cognitive = self.c1 * r1 * (self.particles[i].best_position - self.particles[i].position)
        social = self.c2 * r2 * (self.global_best_position - self.particles[i].position)

        self.particles[i].velocity = (self.w * self.particles[i].velocity + cognitive + social)

        # Update Position
        self.particles[i].position = self.particles[i].position + self.particles[i].velocity

        # Evaluate Fitness
        parameters = self.particles[i].get_network_params()
        fitness = fitness_function(parameters)
        print(f"Particle {i+1} | Test Loss: {fitness:.4f} | Parameters: {parameters}")

        # Update Personal Best
        if fitness < self.particles[i].best_fitness:
          self.particles[i].best_fitness = fitness
          self.particles[i].best_position = self.particles[i].position.copy()

      # Update Global Best
      # TODO: Make this local best instead!
      for particle in self.particles:
        if particle.best_fitness < self.global_best_fitness:
          self.global_best_fitness = particle.best_fitness
          self.global_best_position = particle.best_position.copy()

    best_position = self.global_best_position
    return best_position
  
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
  
# evalutate fitness of a particle by training a neural net on the parameters
def evaluate_particle(parameters, train_dataset, val_dataset, test_dataset, input_dim, num_classes, epochs=5, overfitting_detection = 10):
  train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, generator=g)
  val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], generator=g)
  test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], generator=g)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=parameters['num_hidden_layers'], hidden_size=parameters['hidden_layer_size'], dropout_rate=parameters['dropout_rate']).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=parameters['momentum'], weight_decay=parameters['weight_decay'])
  criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss is used for multi-class classification problems

  running_mean_loss = 0 #stores average loss over minibatches, so this is mean per epoch
  count = 1 #counts minibatches
  loss_history = np.zeros(overfitting_detection)
  loss_history[loss_history == 0.0] = np.nan #do this to do mean_nan - ignores nan values, if we use zeros instead it would messup the mean

  # Training
  model.train()
  for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for images, labels in train_loader:
      images, labels = images.to(device),  labels.float().to(device)
      optimizer.zero_grad()
      loss = criterion(model(images), labels)
      running_mean_loss = running_mean_loss + (loss.item() - running_mean_loss) / count #calculate mean of loss in the batch
      count += 1
      loss.backward()
      optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for images, labels in val_loader:
        images, labels = images.to(device), labels.float().to(device)
        val_loss += criterion(model(images), labels).item()

    val_loss = val_loss / len(val_loader)
    loss_history[epoch%overfitting_detection] = val_loss

    print("val",val_loss) #debug

    #check overfitting
    mean = np.nanmean(loss_history)
    std = np.nanstd(loss_history)
    if val_loss > mean + std:
        print(f"Overfitting detected at epoch {epoch}, stopping search early.")
        break

  # Testing
  model.eval()
  test_loss = 0.0
  with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        test_loss += criterion(model(images), labels).item()
        acc = eval_model(images,labels,model)

  print("test",acc / len(test_loader)) #debug

  return test_loss / len(test_loader) # right now objective function is just the loss (for testing!)

# Penalize infeasible architectures
# TODO: make penalty proportional to how out of bounds the parameters are instead of a flat 1000
def penalty_function(parameters, search_bounds):
    penalty = 0.0

    parameter_keys = ['num_hidden_layers', 'hidden_layer_size', 'learning_rate', 'momentum', 'batch_size', 'weight_decay', 'dropout_rate']

    for parameter, (low, high) in zip(parameter_keys, search_bounds):
        value = parameters[parameter]
        if not (low <= value < high):
            penalty += 1000

    return penalty

# Define multi objecive functions

def _get_complexity(network_params, train_dataset):
    """#helper function for determining complexity by returning number of params"""
    """network_params = dict of network parameters (ie. from pso.particles.get_network_params)"""
    #NOTE: this only works if number of layers is consistent across each layer

    num_layers = network_params['num_hidden_layers']
    layer_size = network_params['hidden_layer_size']
    len_input = len(train_dataset)
    len_output = len(train_dataset.labels[0])

    #middle calculation refers to weights of hidden layers - different when there are 1,2 or 3+ layers
    middle_calculation = 0
    if num_layers == 1:
        middle_calculation = layer_size
    elif num_layers == 2:
        middle_calculation = layer_size**2
    else:
        middle_calculation = (num_layers-1) * (layer_size**2)

    # this = weights on input + weights in hidden layer + weights on output
    num_weights = ((len_input * layer_size) +
                    middle_calculation +
                    (layer_size * len_output))

    # this = biases in hidden layers + biases in output
    num_biases = ((num_layers * layer_size) +
                  len_output)

    return (num_weights + num_biases)

def _complexity_score(network_params):
    """gets the complexity ranking of the particle by comparing it to the highest possible complexity"""
    highest_score = _get_complexity({'num_hidden_layers': 5, 'hidden_layer_size':1024})
    curr_complexity = _get_complexity(network_params)
    return (curr_complexity/highest_score)

def weighted_sums(alpha,beta,complexity,accuracy):
    """take a weighted sum of the accuracy and model size"""
    """alpha, beta = weights. complexity, accuracy = individual obj. function values."""
    return (alpha*complexity)+(beta*accuracy)

# TODO: Make this our multi-objective function instead of just validation loss
def objective_function(parameters, search_bounds, train_dataset, val_dataset, test_dataset, input_dim, num_classes):
  penalty = penalty_function(parameters, search_bounds)
  if penalty > 0:
    print("Infeasible architecture found!")
    return penalty
  return evaluate_particle(parameters, train_dataset, val_dataset, test_dataset, input_dim, num_classes, epochs=5)