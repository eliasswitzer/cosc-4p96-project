import argparse

import numpy as np
import torch
from torchvision import transforms

from pso import PSO, Particle

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

# Datasets
parser.add_argument('--dataset', metavar='dataset', type=str, required=False, default="chest", choices=["chest", "retina", "blood"], help="The dataset the MLP will be trained on.")

args = parser.parse_args()

print(f"Args: {args}")

# Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
g = torch.Generator().manual_seed(args.seed)

# Load dataset
from medmnist import ChestMNIST, RetinaMNIST, BloodMNIST
if args.dataset == 'chest':
    train_dataset = ChestMNIST(split='train', transform=transforms.ToTensor(), download=True)
    val_dataset = ChestMNIST(split='val', transform=transforms.ToTensor(), download=True)
    test_dataset = ChestMNIST(split='test', transform=transforms.ToTensor(), download=True)
elif args.dataset == 'retina':
    train_dataset = RetinaMNIST(split='train', transform=transforms.ToTensor(), download=True)
    val_dataset = RetinaMNIST(split='val', transform=transforms.ToTensor(), download=True)
    test_dataset = RetinaMNIST(split='test', transform=transforms.ToTensor(), download=True)
else:
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

#test of multiobjective helper functions - find complexity score of random particle
#pso1 = PSO(num_particles=10,search_bounds=search_bounds)
#print(_complexity_score(pso1.particles[1].get_network_params()))
#print(_get_complexity(pso1.particles[1].get_network_params()))

pso = PSO(num_particles=args.particles, search_bounds=search_bounds, w=args.w, c1=args.c1, c2=args.c2)
best_position = pso.optimize(args.iterations, search_bounds, args.patience, train_dataset, val_dataset, test_dataset, input_dim, num_classes, g)

best = Particle(search_bounds)
best.position = best_position
print("Best architecture found:", best.get_network_params())

#debugging accuracy
pso2 = PSO(num_particles=1,search_bounds=search_bounds)
parameters = pso2.particles[0].get_network_params()
print(parameters)

