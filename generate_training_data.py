import numpy as np
from random import randint
import torch
from torchvision import transforms
from medmnist import BloodMNIST
from evaluation import collect_particle_data
from particle import Particle

"""used to generate the data used for training the particle predictor"""
"""just for multi-class"""
"""training particles up to 10 epochs"""

search_bounds = [
    (1, 5), # number of hidden layers
    (16, 512), # number of nodes per layer
    (0.0001, 0.01), # learning rate
    (0.0, 0.99), # momentum
    (16, 256), # batch size
    (0.0, 0.001), # weight decay
    (0.0, 0.5) # dropout rate
]

r = randint(0, 987654321)

#set a random seed
np.random.seed(r)
torch.manual_seed(r)
g = torch.Generator().manual_seed(r)

#get the dataset and some of its properties
train_dataset = BloodMNIST(split='train', transform=transforms.ToTensor(), download=True)
test_dataset = BloodMNIST(split='test', transform=transforms.ToTensor(), download=True)
num_classes = len(train_dataset.info['label'])
input_dim = int(np.prod(train_dataset[0][0].shape))

#get a list of particles and train all of them
particle_list = [Particle(search_bounds=search_bounds) for _ in range(100)]
for particle in particle_list:
    collect_particle_data(particle.get_network_params(),train_dataset,test_dataset,input_dim,num_classes,g)