import numpy as np
import matplotlib.pyplot as plt

def get_payoff_matrix(resource_level):
    value = 6 + 4 * resource_level
    cost = 8 - 2 * resource_level

    CC = value / 2              # cooperator vs cooperator
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

    cooperator_fitnesses = []
    cheater_fitnesses = []
    average_fitnesses = []

    x = x0

    for step in range(steps + 1):
        fitness_cooperator, fitness_cheater = get_fitness(payoff_matrix, x)
        average_fitness = get_average_fitness(payoff_matrix, x)

        cooperator_fitnesses.append(fitness_cooperator)
        cheater_fitnesses.append(fitness_cheater)
        average_fitnesses.append(average_fitness)

        if step < steps:
            x = update_population(payoff_matrix, x, dt)

            times.append((step + 1) * dt)
            cooperator_frequencies.append(x)
            cheater_frequencies.append(1 - x)

    return (
        times,
        cooperator_frequencies,
        cheater_frequencies,
        cooperator_fitnesses,
        cheater_fitnesses,
        average_fitnesses,
        payoff_matrix
    )

def find_equilibrium(payoff_matrix):
    a, b = payoff_matrix[0, 0], payoff_matrix[0, 1]
    c, d = payoff_matrix[1, 0], payoff_matrix[1, 1]

    denominator = a - b - c + d
    numerator = d - b

    if abs(denominator) < 1e-12:
        return None

    x_star = numerator / denominator

    if 0 <= x_star <= 1:
        return x_star

    return None

def classify_stability(payoff_matrix, x_star, eps=1e-4):
    dt = 0.01

    if x_star <= eps:
        left = None
        right = x_star + eps
    elif x_star >= 1 - eps:
        left = x_star - eps
        right = None
    else:
        left = x_star - eps
        right = x_star + eps

    left_direction = None
    right_direction = None

    if left is not None:
        next_left = update_population(payoff_matrix, left, dt)
        left_direction = np.sign(next_left - left)

    if right is not None:
        next_right = update_population(payoff_matrix, right, dt)
        right_direction = np.sign(next_right - right)

    if left_direction is not None and right_direction is not None:
        if left_direction > 0 and right_direction < 0:
            return "stable"
        elif left_direction < 0 and right_direction > 0:
            return "unstable"
        else:
            return "semi-stable or neutral"

    if x_star == 0.0 and right_direction is not None:
        if right_direction < 0:
            return "stable"
        elif right_direction > 0:
            return "unstable"
        else:
            return "neutral"

    if x_star == 1.0 and left_direction is not None:
        if left_direction > 0:
            return "stable"
        elif left_direction < 0:
            return "unstable"
        else:
            return "neutral"

    return "undetermined"

def print_results_table(results_table):
    print()
    print(
        f"{'Run':<6}"
        f"{'Start C':<12}"
        f"{'Start Ch':<12}"
        f"{'End C':<12}"
        f"{'End Ch':<12}"
        f"{'Final Fit C':<14}"
        f"{'Final Fit Ch':<14}"
        f"{'Final Avg Fit':<16}"
    )
    print("-" * 98)

    for row in results_table:
        run_number, start_c, start_ch, end_c, end_ch, final_fit_c, final_fit_ch, final_avg_fit = row
        print(
            f"{run_number:<6}"
            f"{start_c:<12.2f}"
            f"{start_ch:<12.2f}"
            f"{end_c:<12.2f}"
            f"{end_ch:<12.2f}"
            f"{final_fit_c:<14.2f}"
            f"{final_fit_ch:<14.2f}"
            f"{final_avg_fit:<16.2f}"
        )

    print()

def get_resource_levels():
    mode = input(
        "Type 1 to enter resource levels manually, or 2 to generate many evenly spaced resource levels: "
    ).strip()

    if mode == "1":
        values = input(
            "Enter resource levels between 0 and 1, separated by commas: "
        ).strip()
        resource_levels = [float(x) for x in values.split(",")]
    else:
        num_levels = int(input("How many resource levels do you want to test? "))
        resource_levels = list(np.linspace(0.0, 1.0, num_levels))

    return resource_levels

def main():
    print(
        "Welcome to the microbial strategy simulation.\n\n"
        "What you should input:\n"
        "- Enter a positive whole number when asked how many times\n"
        "  the simulation should run for each resource level.\n"
        "- This number tells the program how many different starting\n"
        "  cooperator proportions to test for each resource level.\n"
        "- Then either enter resource levels manually or choose to\n"
        "  generate many evenly spaced resource levels from 0 to 1.\n\n"
        "What the program will output:\n"
        "- A payoff matrix for each resource level.\n"
        "- A predicted interior equilibrium, if one exists.\n"
        "- A stability classification for that equilibrium.\n"
        "- A results table showing the starting and ending values\n"
        "  for each run.\n"
        "- Graphs showing how the cooperator proportion changes\n"
        "  over time.\n"
        "- Graphs showing cooperator fitness, cheater fitness,\n"
        "  and average fitness over time.\n"
        "- A final graph comparing predicted equilibrium against\n"
        "  the average simulated final cooperator proportion.\n\n"
        "How to read the table:\n"
        "- Start C: starting proportion of cooperators.\n"
        "- Start Ch: starting proportion of cheaters.\n"
        "- End C: ending proportion of cooperators.\n"
        "- End Ch: ending proportion of cheaters.\n"
        "- Final Fit C: final fitness of cooperators.\n"
        "- Final Fit Ch: final fitness of cheaters.\n"
        "- Final Avg Fit: final average fitness of the population.\n\n"
        "How to read the graphs:\n"
        "- x0 means the initial proportion of cooperators.\n"
        "- If the cooperator curves move toward the same value,\n"
        "  the system is approaching an equilibrium.\n"
        "- If End C is close to 1, cooperators dominate.\n"
        "- If End C is close to 0, cheaters dominate.\n"
        "- If End C stays between 0 and 1, both strategies coexist.\n"
    )

    total_time = 50
    dt = 0.01

    num_runs = int(input("How many times should the simulation run for each resource level? "))
    resource_levels = get_resource_levels()

    equilibrium_x_values = []
    simulated_final_avg_values = []

    for resource_level in resource_levels:
        print("=" * 60)
        print("Resource level:", f"{resource_level:.2f}")

        payoff_matrix = get_payoff_matrix(resource_level)
        print("Payoff matrix:")
        print(np.round(payoff_matrix, 2))

        equilibrium = find_equilibrium(payoff_matrix)
        if equilibrium is not None:
            stability = classify_stability(payoff_matrix, equilibrium)
            print("Predicted interior equilibrium:", f"{equilibrium:.2f}")
            print("Stability:", stability)
        else:
            print("Predicted interior equilibrium: none")
            print("Stability: boundary outcome only")

        plt.figure(figsize=(8, 5))

        results_table = []
        final_end_c_values = []

        for run in range(num_runs):
            if num_runs == 1:
                x0 = 0.5
            else:
                x0 = 0.1 + run * (0.8 / (num_runs - 1))

            results = run_simulation(resource_level, x0, total_time, dt)
            (
                times,
                cooperator_frequencies,
                cheater_frequencies,
                cooperator_fitnesses,
                cheater_fitnesses,
                average_fitnesses,
                payoff_matrix
            ) = results

            start_c = cooperator_frequencies[0]
            start_ch = cheater_frequencies[0]
            end_c = cooperator_frequencies[-1]
            end_ch = cheater_frequencies[-1]
            final_fit_c = cooperator_fitnesses[-1]
            final_fit_ch = cheater_fitnesses[-1]
            final_avg_fit = average_fitnesses[-1]

            final_end_c_values.append(end_c)

            results_table.append((
                run + 1,
                start_c,
                start_ch,
                end_c,
                end_ch,
                final_fit_c,
                final_fit_ch,
                final_avg_fit
            ))

            plt.plot(times, cooperator_frequencies, label=f"x0 = {x0:.2f}")

        print_results_table(results_table)

        simulated_final_average = float(np.mean(final_end_c_values))
        simulated_final_avg_values.append(simulated_final_average)

        if equilibrium is not None:
            equilibrium_x_values.append(equilibrium)
            print("Predicted equilibrium proportion of cooperators:", f"{equilibrium:.2f}")
        else:
            equilibrium_x_values.append(np.nan)
            print("Predicted equilibrium proportion of cooperators: none")

        print("Average simulated final proportion of cooperators:", f"{simulated_final_average:.2f}")

        if equilibrium is not None:
            print(
                "Difference between prediction and simulated average:",
                f"{abs(equilibrium - simulated_final_average):.2f}"
            )
        print()

        plt.title(f"Cooperator Frequency Over Time, resource = {resource_level:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Proportion of Cooperators")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.show()

        if num_runs == 1:
            x0 = 0.5
        else:
            x0 = 0.1 + (num_runs // 2) * (0.8 / (num_runs - 1))

        results = run_simulation(resource_level, x0, total_time, dt)
        (
            times,
            cooperator_frequencies,
            cheater_frequencies,
            cooperator_fitnesses,
            cheater_fitnesses,
            average_fitnesses,
            payoff_matrix
        ) = results

        plt.figure(figsize=(8, 5))
        plt.plot(times, cooperator_fitnesses, label="Cooperator fitness")
        plt.plot(times, cheater_fitnesses, label="Cheater fitness")
        plt.plot(times, average_fitnesses, label="Average fitness")
        plt.title(f"Fitness Over Time, resource = {resource_level:.2f}, x0 = {x0:.2f}")
        plt.xlabel("Time")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.legend()
        plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(resource_levels, equilibrium_x_values, marker='o', label="Predicted equilibrium")
    plt.plot(resource_levels, simulated_final_avg_values, marker='s', label="Average simulated final proportion")
    plt.title("Equilibrium Cooperator Proportion vs Resource Level")
    plt.xlabel("Resource Level")
    plt.ylabel("Proportion of Cooperators")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
