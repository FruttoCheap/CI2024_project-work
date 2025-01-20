import random
import time
from joblib import Parallel, delayed
from crossover import crossover
from mutations import mutate
from selection import multi_objective_selection
from utils import *
from fitness import get_objectives, clear_cache

def genetic_programming(
        X, y,
        pop_size,
        generations,
        max_depth,
        crossover_rate,
        mutation_rate,
        elitism=True,
        elitism_size=3,
        max_no_improvement=5,
        filename=None,
        verbose=True,
):
    start_time = time.time()
    n_variables = len(X[0])

    # Initialize population
    if filename is not None:
        population = import_individuals_from_file(filename)
    else:
        population = initialize_population(pop_size, n_variables, max_depth)

    if verbose:
        init_pop_time = time.time()
        print(f"Population Init Time: {init_pop_time - start_time:.6f}")

    best_fitness_values = []
    best_individuals = []
    gens_without_improvement = 0

    for gen in range(generations):
        clear_cache()
        init_pop_time = time.time()

        # Evaluate fitness of the population
        objectives = Parallel(n_jobs=-1)(delayed(get_objectives)(ind, X, y) for ind in population)
        objectives_dict = {ind: (mse, complexity) for ind, mse, complexity in sorted(objectives, key=lambda x: (x[1], x[2]))}

        if verbose:
            fitnesses_evaluation_time = time.time()
            print(f"Fitness Evaluation Time: {fitnesses_evaluation_time - init_pop_time:.6f}")

        # Track the best individual
        best_individual = list(objectives_dict.keys())[0]
        best_fitness = objectives_dict[best_individual][0]

        if len(best_fitness_values) > 0 and best_fitness_values[-1] == best_fitness:
            gens_without_improvement += 1
        else:
            gens_without_improvement = 0

        best_fitness_values.append(best_fitness)
        best_individuals.append(best_individual)

        print(f"Generation {gen+1}: Best Fitness = {best_fitness:.6f}")
        if best_fitness < 0.0001:
            return best_individual, best_fitness, best_fitness_values

        # Select individuals for the next generation
        selected = multi_objective_selection(objectives_dict)
        if verbose:
            selection_time = time.time()
            print(f"Selection Time: {selection_time - fitnesses_evaluation_time:.6f}")

        # Create next generation
        next_generation = []

        # Elitism
        if elitism:
            top_individuals = Parallel(n_jobs=-1)(
                delayed(simplify_expression)(pop.clone())
                for pop in sorted(objectives_dict.keys(), key=lambda ind: objectives_dict[ind])[:elitism_size]
            )
            next_generation.extend(top_individuals)

        if gens_without_improvement > max_no_improvement:
            if elitism:
                next_generation.extend(objectives_dict.keys()[elitism_size:pop_size * 0.6])
            else:
                next_generation.extend(objectives_dict.keys()[:pop_size * 0.6])

            next_generation.extend(initialize_population(pop_size - len(next_generation), n_variables, max_depth))
            gens_without_improvement = 0
            continue

        if verbose:
            elitism_time = time.time()
            print(f"Elitism Time: {elitism_time - selection_time:.6f}")

        # Generate offspring
        while len(next_generation) < pop_size:
            if random.random() < crossover_rate:
                parent1, parent2 = random.sample(selected, 2)
                offspring1, offspring2 = crossover(parent1, parent2)
                next_generation.append(simplify_expression(offspring1))
                next_generation.append(simplify_expression(offspring2))
            else:
                individual = random.choice(selected)
                mutated = mutate(individual, max_depth, mutation_rate, n_variables)
                next_generation.append(simplify_expression(mutated))

        if verbose:
            new_generation_time = time.time()
            print(f"New Generation Time: {new_generation_time - elitism_time:.6f}")

        # Trim the population
        next_generation = trim_population(next_generation)
        if verbose:
            trimming_time = time.time()
            print(f"Trimming Time: {trimming_time - new_generation_time:.6f}")

        # Randomly trim the population further
        if random.random() < 0.2:
            if verbose:
                print("Trimming Population")
            next_generation = next_generation[:int(0.6 * pop_size)]

        # Ensure the population size is maintained
        while len(next_generation) < pop_size:
            next_generation.append(generate_random_tree(max_depth, n_variables))

        if verbose:
            random_generation_time = time.time()
            print(f"Random Generation Time: {random_generation_time - trimming_time:.6f}")

        population = next_generation
        save_current_population_as_file(population, f'population_{gen}.txt')

    # Final evaluation of the population
    objectives = Parallel(n_jobs=-1)(delayed(get_objectives)(ind, X, y) for ind in population)
    objectives_dict = {ind: (mse, complexity) for ind, mse, complexity in objectives}
    objectives_dict = {ind: objectives_dict[ind] for ind in sorted(objectives_dict.keys(), key=lambda ind: objectives_dict[ind])}

    best_individual = list(objectives_dict.keys())[0]
    best_fitness = objectives_dict[best_individual][0]

    best_fitness_values.append(best_fitness)
    best_individuals.append(best_individual)

    # Final evaluation
    best_fitness = min(best_fitness_values)
    best_index = best_fitness_values.index(best_fitness)
    best_individual = best_individuals[best_index]

    if verbose:
        print(f"\nBest Overall Fitness: {best_fitness:.6f}")

    return best_individual, best_fitness, best_fitness_values