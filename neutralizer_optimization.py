from services.deserialize_optimization_file_service import deserialize_optimization_file
from services.optimization_preparation_service import (
    ga_preparation,
    optimal_solution,
    lbfgsb_optimization
)

import numpy as np


def neutralizer_optimization(data):

    optimization_input = deserialize_optimization_file(data)

    ga_instance, gene_name, gene_per_neutralizer = ga_preparation(
        optimization_input
    )

    # Run the GA
    ga_instance.run()

    # Get the best GA solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print(f"Best GA solution: {solution}, Fitness: {solution_fitness}")

    # Refine the GA solution using L-BFGS-B
    optimized_solution, optimized_fitness = lbfgsb_optimization(
        optimization_input,
        solution,
        gene_name,
        gene_per_neutralizer
    )

    # Only replace the GA solution if L-BFGS-B improved it
    if optimized_fitness > solution_fitness:

        solution = optimized_solution
        solution_fitness = optimized_fitness

        print(
            f"L-BFGS-B improved solution: "
            f"{solution}, Fitness: {solution_fitness}"
        )

    else:

        print(
            "L-BFGS-B did not improve the GA solution. "
            "Keeping GA solution."
        )

    optimal_receptance, primary_system_receptance, receptances_with_detuning, optimezed_neutralizer = optimal_solution(optimization_input, solution, gene_name, gene_per_neutralizer)
    primary_system_frf = 20 * np.log10(abs(primary_system_receptance))
    composed_system_frf = 20 * np.log10(abs(optimal_receptance))

    # Recompute frequencies based on plot bounds
    frequencies = np.linspace(optimization_input.plot_lower_bound, optimization_input.plot_upper_bound, optimization_input.plot_discretization) / (2 * np.pi)  # Convert from rad/s to Hz

    # Convert flat gene/solution list into structured neutralizer-wise dictionary
    structured_solution = []
    i = 0  # start index for slicing gene_name and solution
    viscoelastic_materials = optimization_input.additional_parameters.viscoelastic_materials

    for idx, genes_in_neutralizer in enumerate(gene_per_neutralizer):
        gene_slice = gene_name[i : i + genes_in_neutralizer]
        value_slice = solution[i : i + genes_in_neutralizer]

        # Determine neutralizer type
        type_index = gene_slice.index("type")
        neutralizer_type = int(value_slice[type_index])

        # Default keys depending on type
        if neutralizer_type == 1:
            keys = ["frequency", "type", "modal_position", "viscoelastic_material"]
        elif neutralizer_type in (0, 2):
            keys = ["frequency", "damp", "type", "modal_position"]
        elif neutralizer_type in (3, 4):
            keys = ["shape_factor", "modal_position", "modal_position_tip", "type"]
        else:
            raise ValueError(f"Unknown neutralizer type {neutralizer_type}")

        neutralizer_dict = {k: v for k, v in zip(gene_slice, value_slice)}

        # Add viscoelastic material properties if needed
        if neutralizer_type == 1:
            visco_index = int(neutralizer_dict.get("viscoelastic_material", -1))
            if 0 <= visco_index < len(viscoelastic_materials):
                neutralizer_dict["viscoelastic_material"] = viscoelastic_materials[visco_index].__dict__
            else:
                raise IndexError(f"Viscoelastic material index {visco_index} out of range.")

        # ✅ Add mass to neutralizer_dict
        try:
            neutralizer_mass = optimezed_neutralizer[idx].mass
        except (IndexError, AttributeError):
            raise ValueError(f"Missing or invalid mass for neutralizer index {idx}")

        neutralizer_dict["mass"] = neutralizer_mass

        structured_solution.append(neutralizer_dict)
        i += genes_in_neutralizer

    for item in receptances_with_detuning:
        r = item["receptance"]
        # Convert complex array to magnitude in dB
        magnitude_db = 20 * np.log10(np.abs(r))
        # Store as a plain Python list
        item["receptance"] = magnitude_db.tolist()

    # Prepare the result dictionary
    result = {"solution": structured_solution, "solutionFitness": solution_fitness, "frequency": frequencies.tolist(), "primary_system_frf": primary_system_frf.tolist(), "composed_system_frf": composed_system_frf.tolist(), "receptances_with_detuning": receptances_with_detuning}

    return result
