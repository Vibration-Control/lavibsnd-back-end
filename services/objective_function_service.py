import numpy as np
import matplotlib.pyplot as plt

def objective_function(optimization_data, plot = False):
    
    number_of_neutralizers = len(optimization_data.neutralizers)
    number_of_modes = len(optimization_data.primary_system_natural_frequencies)
    frequency_discretization = len(optimization_data.frequencies)
    modal_mass_array = np.zeros((number_of_modes, number_of_modes), dtype=complex)
    modal_damp_array = np.zeros((number_of_modes, number_of_modes), dtype=complex)
    composed_system_stiffness = np.zeros((number_of_modes, number_of_modes), dtype=complex)
    receptance = np.zeros(frequency_discretization, dtype=complex)
    
    complex_shear_module_per_neutralizer, frequency_index_per_neutralizer, loss_factor_per_neutralizer = calculate_viscoelastic_properties(optimization_data, number_of_neutralizers)

    for i in range(frequency_discretization):
        for j in range(number_of_modes):
            for k in range(number_of_modes):
                for l in range(number_of_neutralizers):
                    frenquency_ratio = optimization_data.frequencies[i]/optimization_data.neutralizers[l].frequency
                    real_shear_module_ratio = complex_shear_module_per_neutralizer[l][i].real / complex_shear_module_per_neutralizer[l][frequency_index_per_neutralizer[l]].real
                    equivalent_mass, equivalent_damp = equivalent_parameters(optimization_data.neutralizers[l],frenquency_ratio, real_shear_module_ratio, loss_factor_per_neutralizer[l][i])
                    modal_mass_array[j][k] += equivalent_mass * optimization_data.primary_system_modes[j][optimization_data.neutralizers[l].modal_position] * optimization_data.primary_system_modes[k][optimization_data.neutralizers[l].modal_position]
                    modal_damp_array[j][k] += equivalent_damp * optimization_data.primary_system_modes[j][optimization_data.neutralizers[l].modal_position] * optimization_data.primary_system_modes[k][optimization_data.neutralizers[l].modal_position]
        for j in range(number_of_modes):
            primary_system_stiffness = complex(optimization_data.primary_system_natural_frequencies[j]**2 - optimization_data.frequencies[i]**2, optimization_data.primary_system_natural_frequencies[j]**2 * optimization_data.primary_system_modal_damping[j])
            for k in range(number_of_modes):
                modal_stiffness = -optimization_data.frequencies[i]**2 * modal_mass_array[j][k] + complex(0,optimization_data.frequencies[i] * modal_damp_array[j][k]) 
                composed_system_stiffness[j][k] = modal_stiffness
                if (j==k):
                    composed_system_stiffness[j][k] = modal_stiffness + primary_system_stiffness
        inverse_composed_system_matrix = np.linalg.inv(composed_system_stiffness)
        for j in range(number_of_modes):
            for k in range(number_of_modes):
                receptance[i] += inverse_composed_system_matrix [j][k] * optimization_data.primary_system_modes[j][optimization_data.response_node_optimization] * optimization_data.primary_system_modes[k][optimization_data.excitation_node_optimization]

    # Plotting the data
    if(plot):
        plt.plot(optimization_data.frequencies/(2*np.pi),20*np.log10(abs(receptance)))
        plt.show()

    return receptance

def equivalent_parameters(neutralizer, frequency_ratio, real_shear_module_ratio = 0, loss_factor = 0):
    equivalent_mass = 0
    equivalent_damp = 0
    if(neutralizer.type == 1):
        denominator = (frequency_ratio ** 2 - real_shear_module_ratio) ** 2 + (real_shear_module_ratio * loss_factor) ** 2
        equivalent_damp = neutralizer.mass * neutralizer.frequency * real_shear_module_ratio * loss_factor * frequency_ratio ** 3 / denominator
        equivalent_mass = - neutralizer.mass * real_shear_module_ratio * (frequency_ratio ** 2 - real_shear_module_ratio * (1 + loss_factor ** 2)) / denominator
    if(neutralizer.type == 2):
        denominator = (frequency_ratio ** 2. - 1.) ** 2. + (2. * neutralizer.damp * frequency_ratio) ** 2.
        equivalent_damp = neutralizer.mass * neutralizer.frequency * 2. * neutralizer.damp * frequency_ratio ** 4. / denominator
        equivalent_mass = -neutralizer.mass * (frequency_ratio ** 2. - (1. + (2. * neutralizer.damp * frequency_ratio) ** 2.)) / denominator

    return equivalent_mass, equivalent_damp

def calculate_viscoelastic_properties(optimization_data, number_of_neutralizers):

    complex_shear_module_per_neutralizer = []
    frequency_index_per_neutralizer = []
    loss_factor_per_neutralizer = []
    for l in range(number_of_neutralizers):
        complex_shear_module_per_neutralizer.append(optimization_data.complex_shear_moduluses[optimization_data.neutralizers[l].viscoelastic_material])
        frequency_index_per_neutralizer.append(np.abs(optimization_data.frequencies - optimization_data.neutralizers[l].frequency).argmin())
        loss_factor_per_neutralizer.append(optimization_data.complex_shear_moduluses[optimization_data.neutralizers[l].viscoelastic_material].imag / optimization_data.complex_shear_moduluses[optimization_data.neutralizers[l].viscoelastic_material].real)
    
    return complex_shear_module_per_neutralizer, frequency_index_per_neutralizer, loss_factor_per_neutralizer