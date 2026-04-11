import argparse

import numpy as np
import torch
from torchvision import transforms

from pso import PSO, Particle
from evaluation import test_model
from visualizations import plot_fitness, plot_distribution, plot_diversity, plot_pareto

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seed', metavar="seed", type=int, required=True, help="The random seed for reproducibility.")

# PSO Parameters
parser.add_argument('-w', metavar='w', type=float, required=False, default=0.729, help="The inertia term weight for PSO algorithm.")
parser.add_argument('-c1', metavar='c1', type=float, required=False, default=1.49445, help="The cognitive acceleration coefficient for PSO algorithm.")
parser.add_argument('-c2', metavar='c2', type=float, required=False, default=1.49445, help="The social acceleration coefficient for PSO algorithm.")
parser.add_argument('-i', '--iterations', metavar='iterations', type=int, required=False, default=10, help="The number of iterations to run the PSO algorithm for")
parser.add_argument('-np', '--particles', metavar='particles', type=int, required=False, default=10, help="The number of particles to run the PSO algorithm with.")
parser.add_argument('-p', '--patience', metavar='patience', type=int, required=False, default=5, help="The number of iterations to test for fitness stagation for early stopping.")
parser.add_argument('-ns', '--neighborhood_size', metavar='neighborhood_size', type=int, required=False, default=3, help="The neighborhood size for local-best PSO algorithm.")
parser.add_argument('-er', '--elite_ratio', metavar='elite_ratio', required=False, default=0, type = int, help="The probability of selecting an elite particle for initialization.")

# Objective Function Parameters
parser.add_argument('-a', '--alpha', metavar='alpha', type=float, required=False, default=0.7, help="The importance of model performance in particle fitness.")
parser.add_argument('-b', '--beta', metavar='beta', type=float, required=False, default=0.3, help="The importance of model complexity in particle fitness.")

# Datasets
parser.add_argument('--dataset', metavar='dataset', type=str, required=False, default="blood", choices=["chest", "blood"], help="The dataset the MLP will be trained on.")

# Enable Particle Predictor
parser.add_argument('-pred','--use_predictor', metavar = 'use_predictor', type = bool, required = False, default =False, help = "Enable the particle predictor to significantly speed up particle evaluation at the expense of some accuracy (MULTI-CLASS ONLY)." )

# Neural Network Parameters
parser.add_argument('-e', '--epochs', metavar='epochs', type=int, required=False, default=10, help="The number of epochs to train each neural network for.")

# Main Components
parser.add_argument('--visualize', action='store_true', help="If included, displays all visualizations.")
parser.add_argument('--final_test', action='store_true', help="If included, trains the best found architecture for 50 epochs and tests it on the test data.")

args = parser.parse_args()

print(f"Args: {args}")

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
g = torch.Generator().manual_seed(args.seed)

# Load dataset
from medmnist import ChestMNIST, BloodMNIST
if args.dataset == 'chest': # multi-label binary
    train_dataset = ChestMNIST(split='train', transform=transforms.ToTensor(), download=True)
    val_dataset = ChestMNIST(split='val', transform=transforms.ToTensor(), download=True)
    test_dataset = ChestMNIST(split='test', transform=transforms.ToTensor(), download=True)
else: # multi-class
    train_dataset = BloodMNIST(split='train', transform=transforms.ToTensor(), download=True)
    val_dataset = BloodMNIST(split='val', transform=transforms.ToTensor(), download=True)
    test_dataset = BloodMNIST(split='test', transform=transforms.ToTensor(), download=True)

num_classes = len(train_dataset.info['label'])
input_dim = int(np.prod(train_dataset[0][0].shape))

search_bounds = [
    (1, 5), # number of hidden layers
    (16, 512), # number of nodes per layer
    (0.0001, 0.01), # learning rate
    (0.0, 0.99), # momentum
    (16, 256), # batch size
    (0.0, 0.001), # weight decay
    (0.0, 0.5) # dropout rate
]

#do some input checking
single_label = "multi-label" not in (train_dataset.info['task'])
if not single_label and args.use_predictor == True:
    print("Warning: Particle Predictor is only compatible for multi-class datasets and is disabled for multi-label classification.")
    args.use_predictor = not args.use_predictor


pso = PSO(num_particles=args.particles, search_bounds=search_bounds, elite_init_ratio = args.elite_ratio, single_label=single_label, w=args.w, c1=args.c1, c2=args.c2)
best_position, history = pso.optimize(args.iterations, search_bounds, args.patience, args.neighborhood_size, train_dataset, val_dataset, input_dim, num_classes, args.epochs, g, alpha=args.alpha, beta=args.beta)

if args.visualize:
    plot_fitness(history)
    plot_diversity(history)
    plot_distribution(history)
    plot_pareto(history)

best = Particle(search_bounds)
best.position = best_position
best_parameters = best.get_network_params()
print("Best architecture found:", best.get_network_params())

if args.final_test:
    final_f1, _ = test_model(best_parameters=best_parameters, train_dataset=train_dataset, test_dataset=test_dataset, input_dim=input_dim, num_classes=num_classes, g=g)

