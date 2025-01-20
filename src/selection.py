from src.node import Node


def dominates(ind1: tuple[float, float], ind2: tuple[float, float]) -> bool:
    """Check if one individual dominates another."""
    return (ind1[0] <= ind2[0] and ind1[1] <= ind2[1]) and (ind1[0] < ind2[0] or ind1[1] < ind2[1])

def pareto_front(population: list[Node], objectives: list[tuple[float, int]]) -> list[Node]:
    """Find the Pareto front of the population."""
    front = []
    for i, obj1 in enumerate(objectives):
        dominated = False
        for j, obj2 in enumerate(objectives):
            if i != j and dominates(obj2, obj1):
                dominated = True
                break
        if not dominated:
            front.append(population[i])
    return front

def diversity_preserving_selection(front: list[Node], objectives: list[tuple[float, int]], count: int) -> list[Node]:
    """Select individuals preserving diversity using crowding distance."""
    distances = [0] * len(front)

    # Calculate crowding distance for each objective
    for i, obj_values in enumerate(zip(*objectives)):
        sorted_indices = sorted(range(len(front)), key=lambda x: obj_values[x])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
        for j in range(1, len(front) - 1):
            distances[sorted_indices[j]] += obj_values[sorted_indices[j + 1]] - obj_values[sorted_indices[j - 1]]

    # Select individuals with the highest crowding distance
    sorted_indices = sorted(range(len(front)), key=lambda x: distances[x], reverse=True)
    return [front[i] for i in sorted_indices[:count]]

def multi_objective_selection(objectives: dict) -> list[Node]:
    """Perform multi-objective selection using Pareto fronts and diversity preservation."""
    selected = []
    remaining_pop = list(objectives.keys())
    remaining_objectives = list(objectives.values())
    pop_size = len(remaining_pop)

    while len(selected) < pop_size:
        # Find the next Pareto front
        front = pareto_front(remaining_pop, remaining_objectives)

        if len(selected) + len(front) <= pop_size:
            selected.extend(front)
        else:
            # Use diversity-preserving selection if the front size exceeds remaining slots
            selected.extend(diversity_preserving_selection(front, remaining_objectives, pop_size - len(selected)))
            break

        # Remove Pareto front solutions from the remaining population
        for individual in front:
            idx = remaining_pop.index(individual)
            del remaining_pop[idx]
            del remaining_objectives[idx]

    return selected