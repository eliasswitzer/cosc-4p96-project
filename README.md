# COSC 4P96 Project - PSO for Optimizing Topology Design of Neural Networks

Designing effective neural network architectures often requires extensive manual tuning of hyperparameters and structural components, a process that is both time-consuming and computationally expensive. This challenge has motivated the development of automated approaches collectively known as Neural Architecture Search (NAS), which aim to systematically discover high-performing architectures with minimal human intervention.

The objective of this project is to investigate the use of Particle Swarm Optimization (PSO) as a search strategy for NAS. Specifically, we aim to leverage PSO to efficiently explore the architecture space and identify neural network configurations that achieve strong predictive performance while maintaining reasonable model complexity and training cost.

This implementation provides several arguments that can be used to run different experiments and control which parts of the framework are run.

## Parameters

- `-s` `--seed` (int, required) - The random seed for reproducibility

Framework Components

- `--no_main_loop` - If included, skips the main PSO loop (use to directly run other simulations)
- `--visualize` - If included, displays all visualizations (fitness by iteration, diversity by iteration, etc.)
- `--final_test` - If included, trains the best found architecture for 50 epochs and tests it on the test data
- `-pred` `--use_predictor` - Enable the particle predictor to significantly speed up particle evaluation at the expense of some accuracy (MULTI-CLASS ONLY)
- `--ns_sim` - Runs the neighborhood size simulation, testing the effects of different neighborhood sizes
- `--np_sim` - Runs the number of particles simluation, testing the effects of different numbers of particles in PSO

PSO Parameters

- `-w` (float) - The inertia term weight, default=0.729
- `-c1` (float) - The cognitive acceleration coefficient, default=1.49445
- `-c2` (float) - The social acceleration coefficient, default=1.49445
- `-i` `--iterations` (int) - The number of iterations to run the search algorithm for, default=20
- `-np` `--particles` (int) - The number of particles to run PSO with, default=30
- `-p` `--patience` (int) - The number of iterations to test for fitness stagnation for early stopping, default=5
- `-ns` `--neighborhood_size` (int) - The neighborhood size for local-best PSO algorithm, default=3
- `-er` `--elite_ratio` (float) - The probability of selecting an elite particle for initialization, default=0

Objective Function Parameters

- `-a` `--alpha` (float) - The importance of model performance in particle fitness, default=0.7
- `-b` `--beta` (float) - The importasnce of model complexity in particle fitness, default=0.3

Datasets

- `--dataset` (str) - The MedMNIST dataset to use, either chest, blood, tissue, or oct, default="blood"

Neural Network Parameters

- `-e` `--epochs` (int) - The number of epochs to train each candidate architecture for, default=10

## Example Run Script

The command used for one of the main PSO experiments was: `python main.py -s 10 --visualize --final_test`
