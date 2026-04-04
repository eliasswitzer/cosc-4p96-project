import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MLP

class Particle:
  def __init__(self, search_bounds):
    self.search_bounds = search_bounds
    num_dims = len(search_bounds)

    self.position = np.array([np.random.uniform(low, high) for low, high in search_bounds]) # initialize position randomly within bounds
    self.velocity = np.zeros(len(search_bounds)) # initialize velocities to zero

    self.best_position = self.position.copy()
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

  def optimize(self, num_iterations, search_bounds, patience, neighborhood_size, train_dataset, val_dataset, test_dataset, input_dim, num_classes, generator):
    # Evaluate initial population
    print(f"Evaluating Initial Population")
    for i in range(len(self.particles)):
      parameters = self.particles[i].get_network_params()
      fitness = objective_function(parameters, search_bounds, train_dataset, val_dataset, test_dataset, input_dim, num_classes, generator)
      print(f"Particle {i+1} | Validation Loss: {fitness:.4f} | Parameters: {parameters}")

      # Set Initial Personal Best
      self.particles[i].best_fitness = fitness
      self.particles[i].best_position = self.particles[i].position.copy()

      # Set Initial Global Best
      if fitness < self.global_best_fitness:
        self.global_best_position = self.particles[i].position.copy()
        self.global_best_fitness = fitness

    no_improvement_count = 0
    # Main PSO Loop
    for iteration in range(num_iterations):
      print(f"Iteration {iteration + 1}/{num_iterations}")
      current_best_fitness = self.global_best_fitness

      # Update velocities using local best
      for i in range(len(self.particles)):
        # Find neighbor indices of current particle
        neighbor_indices = [(i + j) % len(self.particles) for j in range(-(neighborhood_size // 2), (neighborhood_size // 2) + 1)] # gets indices surrounding index i using neighborhood size (uses % to handle wrap-around)

        # Find which local particle has the best personal fitness
        best_neighbor_idx = neighbor_indices[0]
        best_neighbor_fitness = float('inf')
        for idx in neighbor_indices:
           if self.particles[idx].best_fitness < best_neighbor_fitness:
              best_neighbor_fitness = self.particles[idx].best_fitness
              best_neighbor_idx = idx
        local_best_position = self.particles[best_neighbor_idx].best_position

        # Update Velocity
        r1 = np.random.rand(len(self.particles[i].position))
        r2 = np.random.rand(len(self.particles[i].position))

        cognitive = self.c1 * r1 * (self.particles[i].best_position - self.particles[i].position)
        social = self.c2 * r2 * (local_best_position - self.particles[i].position) # uses lbest

        self.particles[i].velocity = (self.w * self.particles[i].velocity + cognitive + social)

        # Update Position
        self.particles[i].position = self.particles[i].position + self.particles[i].velocity

        # Evaluate Fitness
        parameters = self.particles[i].get_network_params()
        fitness = objective_function(parameters, search_bounds, train_dataset, val_dataset, test_dataset, input_dim, num_classes, generator)
        print(f"Particle {i+1} | Test Loss: {fitness:.4f} | Parameters: {parameters}")

        # Update Personal Best
        if fitness < self.particles[i].best_fitness:
          self.particles[i].best_fitness = fitness
          self.particles[i].best_position = self.particles[i].position.copy()

      # Track Global Best (for reporting/final return)
      for particle in self.particles:
        if particle.best_fitness < self.global_best_fitness:
          self.global_best_fitness = particle.best_fitness
          self.global_best_position = particle.best_position.copy()

      # Early Stopping: Check for fitness stagnation
      if (current_best_fitness - self.global_best_fitness) < 1e-4:
         no_improvement_count += 1
      else:
         no_improvement_count = 0

      if no_improvement_count >= patience:
         print("Fitness improvement has stagnated, stopping early!")
         break

      # Early Stopping: Checking particle distance
      positions = np.array([p.position for p in self.particles])
      swarm_spread = np.mean(np.std(positions, axis=0))
      if swarm_spread < 1e-2:
         print("Swarm has physically converged, stopping early!")
         break

    best_position = self.global_best_position
    return best_position

  # Define multi objecive functions
  def _get_complexity(self, network_params, train_dataset):
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

  def _complexity_score(self, network_params):
      """gets the complexity ranking of the particle by comparing it to the highest possible complexity"""
      highest_score = self._get_complexity({'num_hidden_layers': 5, 'hidden_layer_size':1024})
      curr_complexity = self._get_complexity(network_params)
      return (curr_complexity/highest_score)

"""this function gets f1_score with a given dataset and labels"""
def f1_score(x_set,y_set,model):
  with torch.no_grad(): #this disables gradient calculations

    #sigmoid on each label to get the predictions the model has for each label?
    #crit = nn.Sigmoid()
    outputs = model(x_set)

    #conv to numpy
    numpy_pred = outputs.cpu().numpy()
    y_valid_numpy = y_set.cpu().numpy()

    numpy_pred = np.where(numpy_pred > 0 ,1,0)

    #get tp
    right_guesses = np.where((numpy_pred == y_valid_numpy),1,0) #get all instances where predictions = actual labels
    tp = np.logical_and(right_guesses,y_valid_numpy) #get the predictions where they guess positive (1) label correct
    tp = np.sum(tp.astype(int),axis=1)

    #get fn
    tn = np.logical_not(np.logical_or(numpy_pred,y_valid_numpy)) #get pred where neg (0) label was guessed correct. this is where y_pred=0 and labels = 0, so use nor gate
    tn = np.sum(tn.astype(int),axis=1)

    #false positive - where model guesses positive but its actually negative
    wrong_guesses = np.logical_xor(numpy_pred,y_valid_numpy).astype(int) #instances where predictions != actual labels (1 =wrong)
    fp = np.logical_and(wrong_guesses,numpy_pred) #if model guesses wrong (wrong=1) and pred is 1 (positive), means the label is actually 0 (neg). so when both these are 1 we have a fp
    fp = np.sum(fp.astype(int),axis=1)

    #false negative - model guesses negative but its actually positive
    fn = np.logical_and(wrong_guesses,np.logical_not(numpy_pred)) #if model is wrong, and pred is negative, label is 1. negate the pred label then and it with wrong_guess, this will be one if fn
    fn=  np.sum(fn.astype(int),axis=1)

    #do the calculations if denoms aren't 0, or else just set to 0
    recall = np.where((tp+fn) ==0, 0, tp/(tp+fn))
    precision = np.where((tp+fp)==0, 0, tp/(tp+fp))
    f1 = np.where((precision+recall) ==0, 0, 2*(precision*recall)/(precision+recall) )

    return  f1.mean()

# evalutate fitness of a particle by training a neural net on the parameters
def evaluate_particle(parameters, train_dataset, val_dataset, test_dataset, input_dim, num_classes, g, epochs=5, overfitting_detection = 10):
  train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, generator=g)
  val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], generator=g)
  test_loader = DataLoader(test_dataset, batch_size=parameters['batch_size'], generator=g)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = MLP(input_dim=input_dim, num_classes=num_classes, num_layers=parameters['num_hidden_layers'], hidden_size=parameters['hidden_layer_size'], dropout_rate=parameters['dropout_rate']).to(device)

  optimizer = torch.optim.SGD(model.parameters(), lr=parameters['learning_rate'], momentum=parameters['momentum'], weight_decay=parameters['weight_decay'])
  criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([8])) # BCEWithLogitsLoss is used for multi-class classification problems

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
  total_eval_metric = 0.0
  with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        test_loss += criterion(model(images), labels).item()
        curr_eval_metric = f1_score(images, labels, model)
        total_eval_metric+= curr_eval_metric

        print(model(images)[0])
        print(labels[0])

  print(total_eval_metric)
  avg_test_loss = test_loss / len(test_loader)
  avg_eval_metric = total_eval_metric / len(test_loader)

  print(f"Test Loss: {avg_test_loss:.4f} | Test Evaluation Metric (f1/acc): {avg_eval_metric:.4f}") #debug


  return test_loss / len(test_loader)

# Penalize infeasible architectures (penalty is proportional to the distance it goes outside of the search bounds)
def penalty_function(parameters, search_bounds):
    penalty = 0.0

    parameter_keys = ['num_hidden_layers', 'hidden_layer_size', 'learning_rate', 'momentum', 'batch_size', 'weight_decay', 'dropout_rate']

    for parameter, (low, high) in zip(parameter_keys, search_bounds):
        value = parameters[parameter]
        if value < low:
           penalty += (low - value)
        elif value > high:
           penalty += (value - high)

    return penalty * 100

def weighted_sums(alpha,beta,complexity,eval_metric):
    """take a weighted sum of the eval_metric and model size"""
    """alpha, beta = weights. complexity, eval_metric = individual obj. function values."""
    return (alpha*complexity)+(beta*eval_metric)

# TODO: Make this our multi-objective function instead of just validation loss
def objective_function(parameters, search_bounds, train_dataset, val_dataset, test_dataset, input_dim, num_classes, generator):
  """
  Returns fitness of a particle based on model evaluation metric and model complexity. Constraints values to within the search bounds and applies
  a penalty to particles that go outside of those bounds.
  """
  penalty = penalty_function(parameters, search_bounds)

  clipped_parameters = parameters.copy()
  parameter_keys = ['num_hidden_layers', 'hidden_layer_size', 'learning_rate', 'momentum', 'batch_size', 'weight_decay', 'dropout_rate']
  for parameter, (low, high) in zip(parameter_keys, search_bounds):
     clipped_parameters[parameter] = max(low, min(high, parameters[parameter])) # ensure the value of each parameter is within the search bounds

  fitness = evaluate_particle(clipped_parameters, train_dataset, val_dataset, test_dataset, input_dim, num_classes, epochs=5, g=generator)
  return fitness + penalty

