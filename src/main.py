import random

import numpy as np
from matplotlib import pyplot as plt

from gp import genetic_programming

if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    problem = np.load('data/problem_4.npz')
    X = np.array(problem['x'])
    y = np.array(problem['y'])

    # Run Genetic Programming
    best_expr, best_fit, best_fitness_values = genetic_programming(
        X.T, y,
        pop_size=500,
        generations=1000,
        max_depth=5,
        crossover_rate=0.4,
        mutation_rate=0.8,
        elitism=True,
        elitism_size=10,
        verbose=False,
    )

    print(f"\nBest Expression: {best_expr}")

    # Generate predictions
    y_pred = np.array([best_expr.evaluate(x) for x in X.T])

    # Visualize the results considering X has a variable number of arrays inside it
    if X.shape[0] == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(X[0], y, color='blue', label='Data', alpha=0.6)
        plt.plot(X[0], y_pred, color='red', label='Best GP Expression')
        plt.legend()
        plt.title('Symbolic Regression using Genetic Programming')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    else:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[0], X[1], y, color='blue', label='Data', alpha=0.6)
        ax.plot_trisurf(X[0], X[1], y_pred, color='red', alpha=0.6)
        ax.set_xlabel('x_0')
        ax.set_ylabel('x_1')
        ax.set_zlabel('y')
        ax.set_title('Symbolic Regression using Genetic Programming (3D Visualization)')
        plt.legend()
        plt.title(f'Symbolic Regression using Genetic Programming')
        plt.show()

    # Plot the best fitness values over generations
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_values, color='blue', label='Best Fitness')
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()
    plt.show()
