import numpy as np
import matplotlib.pyplot as plt

def get_payoff_matrix(resource_level):
    value = 6 + 4 * resource_level
    cost = 8 - 2 * resource_level

    CC = value / 2              # cooperator vc cooperator
    CCh = 1                     # cooperator vs cheater
    ChC = value                 # cheater vs cooperator
    ChCh = (value - cost) / 2   # cheater vs cheater

    return np.array([
        [CC, CCh],
        [ChC, ChCh]
    ], dtype=float)

def get_fitness(payoff_matrix, x):
    population_vector = np.array([x, 1 - x], dtype=float)
    fitness_vector = payoff_matrix @ population_vector
    return fitness_vector[0], fitness_vector[1]

def get_average_fitness(payoff_matrix, x):
    population_vector = np.array([x, 1 - x], dtype=float)
    return population_vector @ (payoff_matrix @ population_vector)

def update_population(payoff_matrix, x, dt):
    fitness_cooperator, fitness_cheater = get_fitness(payoff_matrix, x)
    average_fitness = get_average_fitness(payoff_matrix, x)

    x_next = x + dt * x * (fitness_cooperator - average_fitness)

    if x_next < 0:
        x_next = 0
    elif x_next > 1:
        x_next = 1

    return x_next

def run_simulation(resource_level, x0, total_time, dt):
    payoff_matrix = get_payoff_matrix(resource_level)
    steps = int(total_time / dt)

    times = [0]
    cooperator_frequencies = [x0]
    cheater_frequencies = [1 - x0]

    x = x0

    for step in range(steps):
        x = update_population(payoff_matrix, x, dt)

        times.append((step + 1) * dt)
        cooperator_frequencies.append(x)
        cheater_frequencies.append(1 - x)

    return times, cooperator_frequencies, cheater_frequencies, payoff_matrix

def main():
    resource_level = 0.5
    x = 0.6
    dt = 0.01

    payoff_matrix = get_payoff_matrix(resource_level)

    print("Payoff matrix:")
    print(payoff_matrix)

    fitness_cooperator, fitness_cheater = get_fitness(payoff_matrix, x)
    print("Cooperator fitness:", fitness_cooperator)
    print("Cheater fitness:", fitness_cheater)

    average_fitness = get_average_fitness(payoff_matrix, x)
    print("Average fitness:", average_fitness)

    x_next = update_population(payoff_matrix, x, dt)
    print("Old x:", x)
    print("New x:", x_next)

if __name__ == "__main__":
    main()
