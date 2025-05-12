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

    # Convert flat gene/solution list into structured neutralizer-wise dictionary
    structured_solution = []
    i = 0  # start index for slicing gene_name and solution

    for genes_in_neutralizer in gene_per_neutralizer:
        gene_slice = gene_name[i:i + genes_in_neutralizer]
        value_slice = solution[i:i + genes_in_neutralizer]

        # Find the neutralizer type from the value slice (based on position of "type")
        type_index = gene_slice.index("type")
        neutralizer_type = int(value_slice[type_index])

        if neutralizer_type == 1:
            keys = ["frequency", "type", "modal_position", "viscoelastic_material"]
        elif neutralizer_type == 2:
            keys = ["frequency", "damp", "type", "modal_position"]
        elif neutralizer_type == 0:
            keys = ["frequency", "damp", "type", "modal_position"]
        else:
            raise ValueError(f"Unknown neutralizer type {neutralizer_type}")

        # Build dictionary using only relevant keys from gene_slice and value_slice
        neutralizer_dict = {k: v for k, v in zip(gene_slice, value_slice)}
        structured_solution.append(neutralizer_dict)

        i += genes_in_neutralizer  # move to next neutralizer


    # Prepare the result dictionary
    result = {
        "solution": structured_solution,
        "solutionFitness": solution_fitness,
        "frequency": frequencies.tolist(),
        "primary_system_frf": primary_system_frf.tolist(),
        "composed_system_frf": composed_system_frf.tolist(),
        "gene_name": gene_name
    }

    return result
