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

class ViscoelasticMaterial:
    def __init__(self, name: str, TT1: int, TT0: int, GH: int, GL: int, beta: float, FI: float, teta1: float, teta2: float):
        self.name = name
        self.TT1 = TT1
        self.TT0 = TT0
        self.GH = GH
        self.GL = GL
        self.beta = beta
        self.FI = FI
        self.teta1 = teta1
        self.teta2 = teta2

class OptimizationVariables:
    def __init__(self, real: List[RealVariable], integer: List[IntegerVariable]):
        self.real = real
        self.integer = integer

class Neutralizer:
    def __init__(self, type: int, mass: float, optimization_variables: OptimizationVariables):
        self.type = type
        self.mass = mass
        self.optimization_variables = optimization_variables

class GeneticAlgorithm:
    def __init__(self, population_size: int, generations: int, crossover: int, mutation: int):
        self.population_size = population_size
        self.generations = generations
        self.crossover = crossover
        self.mutation = mutation

class InputData:
    def __init__(self, primary_system_natural_frequencies: int,primary_system_modal_damping: int, primary_system_modes: List[List[float]], 
                 excitation_node_optimization: int, response_node_optimization: int, 
                 excitation_node_plot: int, response_node_plot: int, plot_type: int, 
                 neutralizers: List[Neutralizer], viscoelastic_materials: List[ViscoelasticMaterial], objective_function_search_lower_bound: int, 
                 objective_function_search_upper_bound: int, objective_function_search_discretization: int,
                 plot_lower_bound: int, plot_upper_bound: int, plot_discretization: int, 
                 genetic_algorithm: GeneticAlgorithm):
        self.primary_system_natural_frequencies = primary_system_natural_frequencies
        self.primary_system_modal_damping = primary_system_modal_damping
        self.primary_system_modes = primary_system_modes
        self.excitation_node_optimization = excitation_node_optimization
        self.response_node_optimization = response_node_optimization
        self.excitation_node_plot = excitation_node_plot
        self.response_node_plot = response_node_plot
        self.plot_type = plot_type
        self.neutralizers = neutralizers
        self.viscoelastic_materials = viscoelastic_materials
        self.objective_function_search_lower_bound = objective_function_search_lower_bound
        self.objective_function_search_upper_bound = objective_function_search_upper_bound
        self.objective_function_search_discretization = objective_function_search_discretization
        self.plot_lower_bound = plot_lower_bound
        self.plot_upper_bound = plot_upper_bound
        self.plot_discretization = plot_discretization
        self.genetic_algorithm = genetic_algorithm

class NeutralizerParameters:
    def __init__(self, mass: float, type: int = 0, frequency: float = 0.0, damp: float = 0.0, viscoelastic_material: int = 0, modal_position: int = 1):
        self.type = type
        self.frequency = frequency
        self.damp = damp
        self.mass = mass
        self.viscoelastic_material = viscoelastic_material
        self.modal_position = modal_position

class objectiveFunctionInput:
    def __init__(self, frequencies: List[float], complex_shear_moduluses: List[float], neutralizers: List[NeutralizerParameters], primary_system_natural_frequencies: int,primary_system_modal_damping: int, primary_system_modes: List[List[float]], excitation_node_optimization: int, response_node_optimization: int ):
        
        self.frequencies = frequencies
        self.complex_shear_moduluses = complex_shear_moduluses
        self.neutralizers = neutralizers
        self.primary_system_natural_frequencies = primary_system_natural_frequencies
        self.primary_system_modal_damping = primary_system_modal_damping
        self.primary_system_modes = primary_system_modes
        self.excitation_node_optimization = excitation_node_optimization
        self.response_node_optimization = response_node_optimization
