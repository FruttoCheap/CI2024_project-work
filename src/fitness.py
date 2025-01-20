# Cache to store evaluated results
import numpy as np

from node import Node, get_all_nodes

cache = {}

def clear_cache():
    """Clear the cache."""
    cache.clear()

def evaluate_individual(individual, x):
    """Evaluate an individual with caching."""
    key = (individual, tuple(x))
    if key in cache:
        return cache[key]
    result = individual.evaluate(x)
    cache[key] = result
    return result

def get_objectives(individual: Node, X: np.ndarray, y: np.ndarray) -> tuple[Node, float, int]:
    """Calculate the objectives (MSE and complexity) for an individual."""
    # Evaluate predictions for all samples in X
    predictions = np.array([evaluate_individual(individual, tuple(x)) for x in X])
    predictions_clipped = np.clip(predictions, -1e10, 1e10)

    # Calculate mean squared error
    mse = np.mean((y - predictions_clipped) ** 2)

    # Calculate complexity (number of nodes in the tree)
    complexity = len(get_all_nodes(individual))

    return individual, float(mse), complexity