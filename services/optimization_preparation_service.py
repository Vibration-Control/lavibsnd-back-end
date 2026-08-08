import numpy as np
from models import ObjectiveFunctionInput, NeutralizerParameters
from typing import List
import pygad
from services.objective_function_service import objective_function
from services.primary_system_service import primary_system_response
from scipy.optimize import minimize


def prepare_objective_function_input(optimization_data, plot=False):

    frequencies = 0
    if plot:
        frequencies = np.linspace(optimization_data.plot_lower_bound, optimization_data.plot_upper_bound, optimization_data.plot_discretization)
    else:
        frequencies = np.linspace(optimization_data.objective_function_search_lower_bound, optimization_data.objective_function_search_upper_bound, optimization_data.objective_function_search_discretization)

    neutralizers: List[NeutralizerParameters] = []
    for neutralizer in optimization_data.neutralizers:
        neutralizer_parameters = NeutralizerParameters(mass=neutralizer.mass, original_mass=neutralizer.mass, mass_type_user_defined=neutralizer.mass_type_user_defined)
        neutralizers.append(neutralizer_parameters)

    complex_shear_moduluses = []
    for viscoelastic_material in optimization_data.additional_parameters.viscoelastic_materials:
        complex_shear_module = complex_shear_modulus(viscoelastic_material, frequencies)
        complex_shear_moduluses.append(complex_shear_module)

    user_defined_dynamic_stiffnesses = []
    for user_defined_dynamic_stiffness in optimization_data.additional_parameters.user_defined_dynamic_stiffnesses:
        user_defined_dynamic_stiffnesses.append(user_defined_dynamic_stiffness.range)

    excitation_node = 0
    response_node = 0
    if plot:
        excitation_node = optimization_data.excitation_node_plot
        response_node = optimization_data.response_node_plot
    else:
        excitation_node = optimization_data.excitation_node_optimization
        response_node = optimization_data.response_node_optimization

    objective_function_input = ObjectiveFunctionInput(
        frequencies=frequencies,
        complex_shear_moduluses=complex_shear_moduluses,
        user_defined_dynamic_stiffnesses=user_defined_dynamic_stiffnesses,
        neutralizers=neutralizers,
        primary_system_natural_frequencies=optimization_data.primary_system_natural_frequencies,
        primary_system_modal_damping=optimization_data.primary_system_modal_damping,
        primary_system_modes=optimization_data.primary_system_modes,
        objective_function_type=optimization_data.objective_function_type,
        excitation_node_optimization=excitation_node,
        response_node_optimization=response_node,
        objective_function_search_lower_bound=optimization_data.objective_function_search_lower_bound,
        objective_function_search_upper_bound=optimization_data.objective_function_search_upper_bound,
    )

    return objective_function_input


def complex_shear_modulus(viscoelastic_material, frequencies):

    lowerShearModulus = viscoelastic_material.lowerShearModulus
    upperShearModulus = viscoelastic_material.upperShearModulus
    temperatureShiftingFactor = viscoelastic_material.temperatureShiftingFactor
    alfaT = alfa(viscoelastic_material.referenceTemperature, viscoelastic_material.workingTemperature, viscoelastic_material.teta1, viscoelastic_material.teta2)

    numerator = lowerShearModulus + (upperShearModulus * temperatureShiftingFactor) * (1j * alfaT * frequencies) ** viscoelastic_material.fractionalDerivativeParameter
    denominator = 1.0 + temperatureShiftingFactor * (1j * alfaT * frequencies) ** viscoelastic_material.fractionalDerivativeParameter

    complex_shear_modulus = numerator / denominator

    return complex_shear_modulus


def alfa(referenceTemperature, workingTemperature, teta1, teta2):
    deltaT = workingTemperature - referenceTemperature
    alfa = 10.0 ** (-teta1 * deltaT / (teta2 + deltaT))
    return alfa


def compute_neutralizer_mass(neutralizer, optimization_data):
    if neutralizer.mass_type_user_defined:
        return neutralizer.original_mass

    valid_modes = [i for i, freq in enumerate(optimization_data.primary_system_natural_frequencies) if optimization_data.objective_function_search_lower_bound <= freq <= optimization_data.objective_function_search_upper_bound]

    total_mass = 0.0
    for i in valid_modes:
        shape = optimization_data.primary_system_modes[i]
        amplitude = abs(shape[neutralizer.modal_position])
        total_mass += neutralizer.original_mass / amplitude**2

    return total_mass / len(valid_modes) if valid_modes else 0.0


def insert_neutralizers(objective_function_input, ga_variables_values, ga_variables_names, ga_variables_per_neutralizer):
    var_offset = 0

    for neutralizer_index, neutralizer in enumerate(objective_function_input.neutralizers):
        num_vars = ga_variables_per_neutralizer[neutralizer_index]

        for i in range(num_vars):
            var_name = ga_variables_names[var_offset + i]
            var_value = ga_variables_values[var_offset + i]

            if var_name == "frequency":
                setattr(neutralizer, var_name, var_value * 2 * np.pi)
            else:
                setattr(neutralizer, var_name, var_value)

        var_offset += num_vars

        neutralizer.mass = compute_neutralizer_mass(neutralizer, objective_function_input)

    return objective_function_input


def calculate_receptances_with_temperature_detuning(optimization_input, plot_input_with_neutralizers):
    receptances_with_detuning = []

    # Iterate through all detuning temperatures
    for temperature in optimization_input.additional_parameters.temperature_detuning:
        # Work on a fresh copy of the input
        modified_input = ObjectiveFunctionInput(**vars(plot_input_with_neutralizers))

        # For each neutralizer that has a viscoelastic material,
        # recompute its complex shear modulus with the detuned temperature
        complex_shear_mods = list(modified_input.complex_shear_moduluses)

        for neu in modified_input.neutralizers:
            if neu.viscoelastic_material is not None:
                material_index = neu.viscoelastic_material

                viscoelastic = optimization_input.additional_parameters.viscoelastic_materials[material_index]
                original_temperature = viscoelastic.workingTemperature
                viscoelastic.workingTemperature = temperature

                complex_shear_mods[material_index] = complex_shear_modulus(
                    viscoelastic,
                    modified_input.frequencies
                )

                viscoelastic.workingTemperature = original_temperature

        modified_input.complex_shear_moduluses = complex_shear_mods

        # Compute receptance
        receptance, objective = objective_function(modified_input, True)

        # Store result as { "temperature": <val>, "receptance": [...] }
        receptances_with_detuning.append({"temperature": temperature, "receptance": receptance})

    return receptances_with_detuning

def _evaluate_solution(
    optimization_input,
    solution,
    gene_name,
    gene_per_neutralizer
):
    objective_function_input = prepare_objective_function_input(
        optimization_input
    )

    objective_function_input_with_neutralizers = insert_neutralizers(
        objective_function_input,
        solution,
        gene_name,
        gene_per_neutralizer
    )

    receptance, objective = objective_function(
        objective_function_input_with_neutralizers
    )

    return objective

def optimal_solution(optimization_input, solution, gene_name, gene_per_neutralizer):
    # Prepare inputs
    plot_input = prepare_objective_function_input(optimization_input, plot=True)
    plot_input_with_neutralizers = insert_neutralizers(plot_input, solution, gene_name, gene_per_neutralizer)

    # Calculate standard receptances
    composed_system_receptance, objective = objective_function(plot_input_with_neutralizers, True)
    primary_system_receptance = primary_system_response(plot_input_with_neutralizers)

    # Calculate receptances under temperature detuning (only if non-empty)
    receptances_with_detuning = []
    if optimization_input.additional_parameters.temperature_detuning is not None and len(optimization_input.additional_parameters.temperature_detuning) > 0:
        receptances_with_detuning = calculate_receptances_with_temperature_detuning(optimization_input, plot_input_with_neutralizers)

    return composed_system_receptance, primary_system_receptance, receptances_with_detuning, plot_input_with_neutralizers.neutralizers


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
            objective_function_input_with_neutralizers = insert_neutralizers(objective_function_input, solution, gene_name, gene_per_neutralizer)
            receptance, objective = objective_function(objective_function_input_with_neutralizers)
            return objective

        return objective_funtion_wrapper

    # Define GA parameters from the genetic_algorithm object
    num_generations = optimization_input.genetic_algorithm.generations
    sol_per_pop = optimization_input.genetic_algorithm.population_size
    num_parents_mating = sol_per_pop // 2
    crossover_probability = optimization_input.genetic_algorithm.crossover / 100
    mutation_probability = optimization_input.genetic_algorithm.mutation / 100

    # Create an instance of the GA
    ga_instance = pygad.GA(num_generations=num_generations, num_parents_mating=num_parents_mating, fitness_func=objective_funtion_wrapper_factory(objective_function_input, gene_name, gene_per_neutralizer), sol_per_pop=sol_per_pop, num_genes=len(gene_types), gene_type=gene_types, gene_space=gene_space, crossover_probability=crossover_probability, mutation_probability=mutation_probability)

    return ga_instance, gene_name, gene_per_neutralizer

def lbfgsb_optimization(
    optimization_input,
    solution,
    gene_name,
    gene_per_neutralizer
):

    # ---------------------------------------------------------
    # Identify continuous variables and their bounds
    # ---------------------------------------------------------

    continuous_indices = []
    continuous_bounds = []

    gene_index = 0

    for neutralizer in optimization_input.neutralizers:

        # Real variables
        for real_variable in neutralizer.optimization_variables.real:

            continuous_indices.append(gene_index)

            continuous_bounds.append(
                (
                    real_variable.lower_bound,
                    real_variable.upper_bound
                )
            )

            gene_index += 1

        # Integer variables
        for integer_variable in neutralizer.optimization_variables.integer:

            # Integer variables remain fixed
            gene_index += 1

    # Nothing to optimize
    if not continuous_indices:

        fitness = _evaluate_solution(
            optimization_input,
            solution,
            gene_name,
            gene_per_neutralizer
        )

        return solution, fitness

    # ---------------------------------------------------------
    # Initial solution
    # ---------------------------------------------------------

    initial_values = np.array(
        [solution[index] for index in continuous_indices],
        dtype=float
    )

    # ---------------------------------------------------------
    # Objective function for L-BFGS-B
    # ---------------------------------------------------------

    def objective_to_minimize(continuous_values):

        candidate_solution = list(solution)

        for index, value in zip(
            continuous_indices,
            continuous_values
        ):
            candidate_solution[index] = float(value)

        fitness = _evaluate_solution(
            optimization_input,
            candidate_solution,
            gene_name,
            gene_per_neutralizer
        )

        # Rescale objective for numerical stability
        scaled_fitness = fitness * 1e13

        # PyGAD maximizes fitness.
        # scipy.optimize.minimize() minimizes.
        return -scaled_fitness

    # ---------------------------------------------------------
    # Run L-BFGS-B
    # ---------------------------------------------------------

    result = minimize(
        objective_to_minimize,
        initial_values,
        method="L-BFGS-B",
        bounds=continuous_bounds,
        options={
            "maxiter": 500,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
            "maxfun": 5000,
            "finite_diff_rel_step": 1e-5
        }
    )

    # ---------------------------------------------------------
    # Build optimized solution
    # ---------------------------------------------------------

    optimized_solution = list(solution)

    for index, value in zip(
        continuous_indices,
        result.x
    ):
        optimized_solution[index] = float(value)

    # ---------------------------------------------------------
    # Evaluate final solution
    # ---------------------------------------------------------

    optimized_fitness = _evaluate_solution(
        optimization_input,
        optimized_solution,
        gene_name,
        gene_per_neutralizer
    )

    iterations = getattr(result, "nit", 0)
    evaluations = getattr(result, "nfev", 0)

    print(
        f"L-BFGS-B finished: success={result.success}, "
        f"iterations={iterations}, "
        f"evaluations={evaluations}"
    )

    print(
        f"L-BFGS-B result: "
        f"{optimized_solution}, "
        f"Fitness: {optimized_fitness}"
    )

    return optimized_solution, optimized_fitness