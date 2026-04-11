import matplotlib.pyplot as plt
import numpy as np

def plot_fitness(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['best_fitness'], label="Best Fitness", color='blue', linewidth=2)
    plt.plot(history['avg_fitness'], label="Average Swarm Fitness", color='orange', linestyle='--')
    plt.title("PSO Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Maximization)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_diversity(history):
    plt.figure(figsize=(10,5))
    plt.plot(history['diversity'], color='green')
    plt.title("Swarm Diversity Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Swarm Diversity (Mean Distance to Centroid)")
    plt.grid(True)
    plt.show()

def plot_distribution(history):
    architectures = history['architectures']
    layers = [a['num_hidden_layers'] for a in architectures]
    hidden_size = [a['hidden_layer_size'] for a in architectures]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(layers, bins=range(1, 7), color='blue')
    axes[0].set_title("Distribution of Hidden Layers")

    axes[1].hist(hidden_size, bins=10, color='orange')
    axes[1].set_title("Distribution of Hidden Layer Sizes")
    plt.show()

def plot_pareto(history):
    points = np.array(history['pareto_data'])
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 1], points[:, 0])
    plt.xlabel("Complexity")
    plt.ylabel("Performance")
    plt.grid(True)
    plt.show()