import numpy as np

def primary_system_response(optimization_data):
    number_of_modes = len(optimization_data.primary_system_natural_frequencies)
    frequency_discretization = len(optimization_data.frequencies)
    
    composed_system_stiffness = np.zeros((number_of_modes, number_of_modes), dtype=complex)
    receptance = np.zeros(frequency_discretization, dtype=complex)

    for i in range(frequency_discretization):
        # Initialize the modal stiffness matrix for this frequency point
        for j in range(number_of_modes):
            for k in range(number_of_modes):
                if j == k:
                    # Only diagonal terms: primary system dynamic stiffness
                    primary_system_stiffness = complex(
                        optimization_data.primary_system_natural_frequencies[j]**2 - optimization_data.frequencies[i]**2,
                        optimization_data.primary_system_natural_frequencies[j]**2 * optimization_data.primary_system_modal_damping[j]
                    )
                    composed_system_stiffness[j][k] = primary_system_stiffness
                else:
                    # No coupling between modes (diagonal system)
                    composed_system_stiffness[j][k] = 0.0

        # Invert the diagonal matrix (can use full inversion for generality)
        inverse_composed_system_matrix = np.linalg.inv(composed_system_stiffness)

        # Assemble the receptance at this frequency
        for j in range(number_of_modes):
            for k in range(number_of_modes):
                receptance[i] += (
                    inverse_composed_system_matrix[j][k]
                    * optimization_data.primary_system_modes[j][optimization_data.response_node_optimization]
                    * optimization_data.primary_system_modes[k][optimization_data.excitation_node_optimization]
                )

    return receptance
