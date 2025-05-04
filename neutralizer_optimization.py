from services.deserialize_optimization_file_service import deserialize_optimization_file
from services.optimization_preparation_service import ga_preparation, optimal_solution

import numpy as np

def neutralizer_optimization(data):
    optimization_input = deserialize_optimization_file(data)
    ga_instance, gene_name, gene_per_neutralizer = ga_preparation(optimization_input)
    
    # Run the GA
    ga_instance.run()

    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {solution}, Fitness: {solution_fitness}")
    
    optimal_receptance, primary_system_receptance = optimal_solution(optimization_input, solution, gene_name, gene_per_neutralizer)
    primary_system_frf = 20 * np.log10(abs(primary_system_receptance))
    composed_system_frf = 20 * np.log10(abs(optimal_receptance))

    # Recompute frequencies based on plot bounds
    frequencies = np.linspace(
        optimization_input.plot_lower_bound,
        optimization_input.plot_upper_bound,
        optimization_input.plot_discretization
    ) / (2 * np.pi)  # Convert from rad/s to Hz

    # Prepare the result dictionary
    result = {
        "solution": solution.tolist(),  # Convert numpy array to list
        "solutionFitness": solution_fitness,
        "frequency": frequencies.tolist(),  # Use recomputed frequencies
        "primary_system_frf": primary_system_frf.tolist(),
        "composed_system_frf": composed_system_frf.tolist()
    }

    return result
