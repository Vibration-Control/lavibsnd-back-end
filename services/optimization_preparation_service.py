import numpy as np
from models import objectiveFunctionInput,NeutralizerParameters
from typing import List
import pygad
from services.objective_function_service import objective_function


def prepare_objective_function_input(optimization_data):

    frequencies = np.linspace(optimization_data.objective_function_search_lower_bound, optimization_data.objective_function_search_upper_bound, optimization_data.objective_function_search_discretization)
    neutralizers: List[NeutralizerParameters] = []
    for neutralizer in optimization_data.neutralizers:
        neutralizer_parameters = NeutralizerParameters(
            mass = neutralizer.mass
        )
        neutralizers.append(neutralizer_parameters)

    complex_shear_moduluses = []
    for viscoelastic_material in optimization_data.viscoelastic_materials:
        complex_shear_module = complex_shear_modulus(viscoelastic_material, frequencies)
        complex_shear_moduluses.append(complex_shear_module)

    objective_function_input = objectiveFunctionInput(
        frequencies = frequencies,
        complex_shear_moduluses = complex_shear_moduluses,
        neutralizers = neutralizers,
        primary_system_natural_frequencies = optimization_data.primary_system_natural_frequencies,
        primary_system_modal_damping = optimization_data.primary_system_modal_damping,
        primary_system_modes = optimization_data.primary_system_modes,
        excitation_node_optimization = optimization_data.excitation_node_optimization,
        response_node_optimization = optimization_data.response_node_optimization
    )

    return objective_function_input

def complex_shear_modulus(viscoelastic_material, frequencies):

    GL = viscoelastic_material.GL
    GH = viscoelastic_material.GH
    FI = viscoelastic_material.FI
    alfaT = alfa(viscoelastic_material.TT0, viscoelastic_material.TT1, viscoelastic_material.teta1, viscoelastic_material.teta2)
    
    numerator = GL + (GH * FI) * (1j * alfaT * frequencies) ** viscoelastic_material.beta
    denominator = 1.0 + FI * (1j * alfaT * frequencies) ** viscoelastic_material.beta
    
    complex_shear_modulus = numerator / denominator

    return complex_shear_modulus

def alfa(TT0, TT1, teta1, teta2):
    deltaT = TT1 - TT0
    alfa = 10.0 ** (-teta1 * deltaT / (teta2 + deltaT))
    return alfa

def insert_neutralizers(objective_function_input,ga_variables_values,ga_variables_names, ga_variables_per_neutralizer):
    for neutralizer_index, neutralizer in enumerate(objective_function_input.neutralizers):
        for variable_index in range(ga_variables_per_neutralizer[neutralizer_index]):
            if ga_variables_names[variable_index] == "frequency":
                setattr(neutralizer, ga_variables_names[variable_index], ga_variables_values[variable_index]*2*np.pi)
            else:
                setattr(neutralizer, ga_variables_names[variable_index], ga_variables_values[variable_index])

    return objective_function_input

def prepare_plot_input(optimization_data):

    frequencies = np.linspace(optimization_data.plot_lower_bound, optimization_data.plot_upper_bound, optimization_data.plot_discretization)
    neutralizers: List[NeutralizerParameters] = []
    for neutralizer in optimization_data.neutralizers:
        neutralizer_parameters = NeutralizerParameters(
            mass = neutralizer.mass
        )
        neutralizers.append(neutralizer_parameters)

    complex_shear_moduluses = []
    for viscoelastic_material in optimization_data.viscoelastic_materials:
        complex_shear_module = complex_shear_modulus(viscoelastic_material, frequencies)
        complex_shear_moduluses.append(complex_shear_module)

    plot_input = objectiveFunctionInput(
        frequencies = frequencies,
        complex_shear_moduluses = complex_shear_moduluses,
        neutralizers = neutralizers,
        primary_system_natural_frequencies = optimization_data.primary_system_natural_frequencies,
        primary_system_modal_damping = optimization_data.primary_system_modal_damping,
        primary_system_modes = optimization_data.primary_system_modes,
        excitation_node_optimization = optimization_data.excitation_node_plot,
        response_node_optimization = optimization_data.response_node_plot
    )

    return plot_input

def optimal_solution(optimization_input, solution, gene_name, gene_per_neutralizer):
    plot_input = prepare_plot_input(optimization_input)
    plot_input_with_neutralizers = insert_neutralizers(plot_input, solution, gene_name, gene_per_neutralizer)
    receptance = objective_function(plot_input_with_neutralizers, True)
    return receptance

def ga_preparation(optimization_input):
    objective_function_input = prepare_objective_function_input(optimization_input)

    # Initialize the arrays
    gene_types = []
    gene_space = []
    gene_name = []
    gene_per_neutralizer = []

    # Iterate through each neutralizer
    for neutralizer in optimization_input.neutralizers:
        gene_per_neutralizer_count = 0
        # Handle real variables
        for real_variable in neutralizer.optimization_variables.real:
            gene_types.append(float)
            lower_bound = real_variable.lower_bound
            upper_bound = real_variable.upper_bound
            discretization = real_variable.discretization
            gene_space.append(np.linspace(lower_bound, upper_bound, discretization).tolist())
            gene_name.append(real_variable.name)
            gene_per_neutralizer_count += 1

        
        # Handle integer variables
        for integer_variable in neutralizer.optimization_variables.integer:
            gene_types.append(int)
            gene_space.append(integer_variable.range)
            gene_name.append(integer_variable.name)
            gene_per_neutralizer_count += 1

        gene_per_neutralizer.append(gene_per_neutralizer_count)


    def objective_funtion_wrapper_factory(objective_function_input, gene_name, gene_per_neutralizer):

        def objective_funtion_wrapper(ga_instance, solution, solution_idx):
            objective_function_input_with_neutralizers = insert_neutralizers(objective_function_input,solution, gene_name, gene_per_neutralizer)
            receptance = objective_function(objective_function_input_with_neutralizers)
            return -np.sum(receptance * np.conj(receptance)).real
        
        return objective_funtion_wrapper

    # Define GA parameters from the genetic_algorithm object
    num_generations = optimization_input.genetic_algorithm.generations
    sol_per_pop = optimization_input.genetic_algorithm.population_size
    num_parents_mating = sol_per_pop // 2
    crossover_probability = optimization_input.genetic_algorithm.crossover / 100
    mutation_probability = optimization_input.genetic_algorithm.mutation / 100

    # Create an instance of the GA
    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=objective_funtion_wrapper_factory(objective_function_input, gene_name, gene_per_neutralizer),
                        sol_per_pop=sol_per_pop,
                        num_genes=len(gene_types),
                        gene_type=gene_types,
                        gene_space=gene_space,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability)

    return ga_instance,gene_name,gene_per_neutralizer