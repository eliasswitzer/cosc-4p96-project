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
    points = np.array(history['pareto_data'][-1]) # get points from final iteration

    pareto = []
    for i, (p_i, c_i) in enumerate(points):
        dominated = False
        for j, (p_j , c_j) in enumerate(points):
            if i == j:
                continue
            if (p_j <= p_i and c_j <= c_i) and (p_j < p_i or c_j < c_i):
                dominated = True
                break
        if not dominated:
            pareto.append((p_i, c_i))
    
    pareto = np.array(pareto)

    if len(pareto) > 0:
        pareto = pareto[np.argsort(pareto[:, 1])]

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 1], points[:, 0], alpha=0.3, label="All Solutions")

    if len(pareto) > 0:
        plt.scatter(pareto[:, 1], pareto[:, 0], color='red', label="Pareto Front")

    plt.xlabel("Complexity")
    plt.ylabel("Performance")
    plt.title("Pareto Front (Non-Dominated Solutions)")
    plt.legend()
    plt.grid(True)
    plt.show()
