from typing import List


class IntegerVariable:
    def __init__(self, name: str, range: List[int]):
        self.name = name
        self.range = range


class RealVariable:
    def __init__(self, name: str, lower_bound: float, upper_bound: float, discretization: int):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.discretization = discretization


class DynamicStiffnessVariable:
    def __init__(self, name: str, range: List[float]):
        self.name = name
        self.range = range


class ViscoelasticMaterial:
    def __init__(self, name: str, workingTemperature: int, referenceTemperature: int, upperShearModulus: int, lowerShearModulus: int, fractionalDerivativeParameter: float, temperatureShiftingFactor: float, teta1: float, teta2: float):
        self.name = name
        self.workingTemperature = workingTemperature
        self.referenceTemperature = referenceTemperature
        self.upperShearModulus = upperShearModulus
        self.lowerShearModulus = lowerShearModulus
        self.fractionalDerivativeParameter = fractionalDerivativeParameter
        self.temperatureShiftingFactor = temperatureShiftingFactor
        self.teta1 = teta1
        self.teta2 = teta2


class OptimizationVariables:
    def __init__(self, real: List[RealVariable], integer: List[IntegerVariable]):
        self.real = real
        self.integer = integer


class Neutralizer:
    def __init__(self, type: int, mass: float, mass_type_user_defined: bool, optimization_variables: OptimizationVariables):
        self.type = type
        self.mass = mass
        self.mass_type_user_defined = mass_type_user_defined
        self.optimization_variables = optimization_variables


class GeneticAlgorithm:
    def __init__(self, population_size: int, generations: int, crossover: int, mutation: int):
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover
        self.mutation = mutation


class AdditionalParameters:
    def __init__(self, user_defined_dynamic_stiffnesses: List[DynamicStiffnessVariable], temperature_detuning: List[float], viscoelastic_materials: List[ViscoelasticMaterial]):
        self.user_defined_dynamic_stiffnesses = user_defined_dynamic_stiffnesses
        self.temperature_detuning = temperature_detuning
        self.viscoelastic_materials = viscoelastic_materials


class InputData:
    def __init__(
        self,
        primary_system_natural_frequencies: int,
        primary_system_modal_damping: int,
        primary_system_modes: List[List[float]],
        excitation_node_optimization: int,
        response_node_optimization: int,
        excitation_node_plot: int,
        response_node_plot: int,
        objective_function_type: int, 
        plot_type: int,
        neutralizers: List[Neutralizer],
        additional_parameters: AdditionalParameters,
        objective_function_search_lower_bound: int,
        objective_function_search_upper_bound: int,
        objective_function_search_discretization: int,
        plot_lower_bound: int,
        plot_upper_bound: int,
        plot_discretization: int,
        genetic_algorithm: GeneticAlgorithm,
    ):
        self.primary_system_natural_frequencies = primary_system_natural_frequencies
        self.primary_system_modal_damping = primary_system_modal_damping
        self.primary_system_modes = primary_system_modes
        self.excitation_node_optimization = excitation_node_optimization
        self.response_node_optimization = response_node_optimization
        self.excitation_node_plot = excitation_node_plot
        self.response_node_plot = response_node_plot
        self.objective_function_type = objective_function_type
        self.plot_type = plot_type
        self.neutralizers = neutralizers
        self.additional_parameters = additional_parameters
        self.objective_function_search_lower_bound = objective_function_search_lower_bound
        self.objective_function_search_upper_bound = objective_function_search_upper_bound
        self.objective_function_search_discretization = objective_function_search_discretization
        self.plot_lower_bound = plot_lower_bound
        self.plot_upper_bound = plot_upper_bound
        self.plot_discretization = plot_discretization
        self.genetic_algorithm = genetic_algorithm


class NeutralizerParameters:
    def __init__(self, mass: float = 0.1, original_mass: float = 0.1, mass_type_user_defined: bool = True, type: int = 0, frequency: float = 1.0, damp: float = 0.0, viscoelastic_material: int = 0, dynamic_stiffness: int = 0, modal_position: int = 1, modal_position_tip: int = 1, shape_factor: float = 1.0):
        self.type = type
        self.frequency = frequency
        self.damp = damp
        self.mass = mass
        self.original_mass = original_mass
        self.mass_type_user_defined = mass_type_user_defined
        self.viscoelastic_material = viscoelastic_material
        self.dynamic_stiffness = dynamic_stiffness
        self.modal_position = modal_position
        self.modal_position_tip = modal_position_tip
        self.shape_factor = shape_factor


class ObjectiveFunctionInput:
    def __init__(self, frequencies: list[float], user_defined_dynamic_stiffnesses: list[float], complex_shear_moduluses: list[float], neutralizers: list[NeutralizerParameters], primary_system_natural_frequencies: list[float], primary_system_modal_damping: list[float], primary_system_modes: list[list[float]],objective_function_type:int, excitation_node_optimization: int, response_node_optimization: int, objective_function_search_lower_bound: float, objective_function_search_upper_bound: float):
        self.frequencies = frequencies
        self.complex_shear_moduluses = complex_shear_moduluses
        self.user_defined_dynamic_stiffnesses = user_defined_dynamic_stiffnesses
        self.neutralizers = neutralizers
        self.primary_system_natural_frequencies = primary_system_natural_frequencies
        self.primary_system_modal_damping = primary_system_modal_damping
        self.primary_system_modes = primary_system_modes
        self.objective_function_type = objective_function_type
        self.excitation_node_optimization = excitation_node_optimization
        self.response_node_optimization = response_node_optimization
        self.objective_function_search_lower_bound = objective_function_search_lower_bound
        self.objective_function_search_upper_bound = objective_function_search_upper_bound
