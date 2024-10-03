import json
from typing import List
import numpy as np
from models import InputData, IntegerVariable, RealVariable, DynamicStiffnessVariable, OptimizationVariables, Neutralizer, GeneticAlgorithm, ViscoelasticMaterial, AdditionalParameters  # Adjust import path as needed

def deserialize_optimization_file(json_data) -> InputData:
    json_payload_str = json.dumps(json_data)
    data = json.loads(json_payload_str)

    def parse_integer_variable(item):
        return IntegerVariable(name=item['name'], range=item['range'])

    def parse_real_variable(item):
        return RealVariable(
            name=item['name'],
            lower_bound=item['lowerBound'],
            upper_bound=item['upperBound'],
            discretization=item['discretization']
        )

    def parse_dynamic_stiffness_variable(item):
        return DynamicStiffnessVariable(name=item['name'], range=item['range'])

    def parse_viscoelastic_material(item):
        return ViscoelasticMaterial(
            name=item['name'],
            TT1=item.get('TT1', 0),
            TT0=item.get('TT0', 0),
            GH=item.get('GH', 0),
            GL=item.get('GL', 0),
            beta=item.get('beta', 0.0),
            FI=item.get('FI', 0.0),
            teta1=item.get('teta1', 0.0),
            teta2=item.get('teta2', 0.0)
        )

    def parse_optimization_variables(data):
        real = [parse_real_variable(item) for item in data['real']]
        integer = [parse_integer_variable(item) for item in data['integer']]
        return OptimizationVariables(real=real, integer=integer)

    def parse_neutralizer(item):
        optimization_variables = parse_optimization_variables(item['optimizationVariables'])
        return Neutralizer(type=item.get('type', 0), mass=item['mass'], optimization_variables=optimization_variables)

    def parse_genetic_algorithm(data):
        return GeneticAlgorithm(
            population_size=data['populationSize'],
            generations=data['generations'],
            crossover=data['crossover'],
            mutation=data['mutation']
        )

    def parse_additional_parameters(data):
        user_defined_dynamic_stiffnesses = [parse_dynamic_stiffness_variable(item) for item in data['userDefinedDynamicStiffnesses']]
        viscoelastic_materials = [parse_viscoelastic_material(item) for item in data['viscoelasticMaterials']]
        return AdditionalParameters(user_defined_dynamic_stiffnesses=user_defined_dynamic_stiffnesses, viscoelastic_materials=viscoelastic_materials)

    # Parse neutralizers
    neutralizers = [parse_neutralizer(item) for item in data['neutralizers']]

    # Parse additional parameters
    additional_parameters = parse_additional_parameters(data['additionalParameters'])

    # Parse genetic algorithm
    genetic_algorithm = parse_genetic_algorithm(data['geneticAlgorithm'])

    input_data = InputData(
        primary_system_natural_frequencies=np.array(data['primarySystemNaturalFrequencies'])*2*np.pi,
        primary_system_modal_damping=data['primarySystemModalDamping'],
        primary_system_modes=data['primarySystemModes'],
        excitation_node_optimization=data['excitationNodeOptimization'],
        response_node_optimization=data['responseNodeOptimization'],
        excitation_node_plot=data['excitationNodePlot'],
        response_node_plot=data['responseNodePlot'],
        plot_type=data['plotType'],
        neutralizers=neutralizers,
        additional_parameters=additional_parameters,
        objective_function_search_lower_bound=data['objectiveFunctionSearchLowerBound']*2*np.pi,
        objective_function_search_upper_bound=data['objectiveFunctionSearchUpperBound']*2*np.pi,
        objective_function_search_discretization=data['objectiveFunctionSearchDiscretization'],
        plot_lower_bound=data['plotLowerBound']*2*np.pi,
        plot_upper_bound=data['plotUpperBound']*2*np.pi,
        plot_discretization=data['plotDiscretization'],
        genetic_algorithm=genetic_algorithm
    )

    return input_data
