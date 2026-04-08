import numpy as np
import random

from particle import Particle
from evaluation import evaluate_particle

from data import best_architectures_multiclass, best_architectures_multilabel

class PSO:
  def __init__(self, num_particles, search_bounds, single_label, elite_init_ratio=0, w=0.729, c1=1.49445, c2=1.49445):

    #initialize particles - either randomly or load already good solutions based on elite_init_ratio
    self.particles =[]
    for i in range(num_particles):
        if random.randint(1,100) < elite_init_ratio: # elite particle
          if single_label:
            self.particles.append(Particle(search_bounds=search_bounds,init=random.choice(best_architectures_multiclass)))
          else:
            self.particles.append(Particle(search_bounds=search_bounds,init=random.choice(best_architectures_multilabel)))
        else: # random
          self.particles.append(Particle(search_bounds=search_bounds))

    self.v_max = np.array([(high - low) * 0.2 for low, high in search_bounds])
    self.v_min = -self.v_max

    self.global_best_position = None
    self.global_best_fitness = float('inf')
    self.w, self.c1, self.c2, = w, c1, c2

  def optimize(self, num_iterations, search_bounds, patience, neighborhood_size, train_dataset, val_dataset, input_dim, num_classes, epochs, generator, alpha, beta):
    # Evaluate initial population
    print(f"Evaluating Initial Population")
    for i in range(len(self.particles)):
      parameters = self.particles[i].get_network_params()
      fitness = self.objective_function(num_iterations, parameters, search_bounds, train_dataset, val_dataset, input_dim, num_classes, epochs, generator, alpha, beta)
      print(f"Particle {i+1} | Fitness: {fitness:.4f} | Parameters: {parameters}")

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

        new_velocity = (self.w * self.particles[i].velocity + cognitive + social)

        self.particles[i].velocity = np.clip(new_velocity, self.v_min, self.v_max) # Apply velocity clamping

        # Update Position
        self.particles[i].position = self.particles[i].position + self.particles[i].velocity

        # Evaluate Fitness
        parameters = self.particles[i].get_network_params()
        fitness = self.objective_function(num_iterations, parameters, search_bounds, train_dataset, val_dataset, input_dim, num_classes, epochs, generator, alpha, beta)
        print(f"Particle {i+1} | Fitness: {fitness:.4f} | Parameters: {parameters}")

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
  def _get_complexity(self, network_params, input_dim, num_classes):
      """#helper function for determining complexity by returning number of params"""
      """network_params = dict of network parameters (ie. from pso.particles.get_network_params)"""
      #NOTE: this only works if number of layers is consistent across each layer

      num_layers = network_params['num_hidden_layers']
      layer_size = network_params['hidden_layer_size']

      if num_layers == 1:
         middle_weights = 0
      else:
         middle_weights = (num_layers - 1) * (layer_size  ** 2)

      # this = weights on input + weights in hidden layer + weights on output
      num_weights = (input_dim * layer_size) + middle_weights + (layer_size * num_classes)

      # this = biases in hidden layers + biases in output
      num_biases = (num_layers * layer_size) + num_classes

      return num_weights + num_biases

  def get_complexity_score(self, network_params, search_bounds, input_dim, num_classes):
      """gets the complexity ranking of the particle by comparing it to the highest possible complexity"""
      complexity = self._get_complexity(network_params, input_dim, num_classes)
      max_layers = int(np.round(search_bounds[0][1]))
      max_nodes = int(np.round(search_bounds[1][1]))
      max_params = self._get_complexity({'num_hidden_layers': max_layers, 'hidden_layer_size': max_nodes}, input_dim, num_classes)


      #return np.log10(complexity) / np.log10(max_params) # adding log here to make penalty less aggressive for larger models
      return complexity / max_params

  # Penalize infeasible architectures (penalty is proportional to the distance it goes outside of the search bounds)
  def penalty_function(self, parameters, search_bounds):
      penalty = 0.0

      parameter_keys = ['num_hidden_layers', 'hidden_layer_size', 'learning_rate', 'momentum', 'batch_size', 'weight_decay', 'dropout_rate']

      for parameter, (low, high) in zip(parameter_keys, search_bounds):
          value = parameters[parameter]
          if value < low:
            penalty += (low - value)
          elif value > high:
            penalty += (value - high)

      return penalty * 100

  def objective_function(self, num_iterations, parameters, search_bounds, train_dataset, val_dataset, input_dim, num_classes, epochs, generator, alpha=0.7, beta=0.3):
    """
    Returns fitness of a particle based on model evaluation metric and model complexity. Constrains values to within the search bounds and applies
    a penalty to particles that go outside of those bounds.
    """
    penalty = self.penalty_function(parameters, search_bounds)

    clipped_parameters = parameters.copy()
    parameter_keys = ['num_hidden_layers', 'hidden_layer_size', 'learning_rate', 'momentum', 'batch_size', 'weight_decay', 'dropout_rate']
    for parameter, (low, high) in zip(parameter_keys, search_bounds):
      clipped_parameters[parameter] = max(low, min(high, parameters[parameter])) # ensure the value of each parameter is within the search bounds

    # Model Performance
    eval_metric1,eval_metric2 = evaluate_particle(clipped_parameters, train_dataset, val_dataset, input_dim, num_classes, epochs=epochs, g=generator) #for multi-class: (acc,acc) multi-label: (f1,ham)

    # Model Complexity
    complexity = self.get_complexity_score(clipped_parameters, search_bounds, input_dim, num_classes)

    # Maximizing performance, minimizing model complexity
    print(f"Performance: {eval_metric2} | Complexity: {(1-complexity)} | Penalty: {penalty}") # debug
    fitness = (alpha * (eval_metric2)) + (beta * (1-complexity))
    return fitness - penalty

