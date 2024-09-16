import json
from typing import List
import numpy as np
from models import InputData, IntegerVariable, RealVariable, OptimizationVariables, Neutralizer, GeneticAlgorithm, ViscoelasticMaterial  # Adjust import path as needed

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

    def parse_viscoelastic_material(item):
        return ViscoelasticMaterial(
            name=item['name'],
            TT1=item['TT1'],
            TT0=item['TT0'],
            GH=item['GH'],
            GL=item['GL'],
            beta=item['beta'],
            FI=item['FI'],
            teta1=item['teta1'],
            teta2=item['teta2']
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

    neutralizers = [parse_neutralizer(item) for item in data['neutralizers']]
    viscoelastic_materials = [parse_viscoelastic_material(item) for item in data['viscoelasticMaterials']]
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
        viscoelastic_materials=viscoelastic_materials,
        objective_function_search_lower_bound=data['objectiveFunctionSearchLowerBound']*2*np.pi,
        objective_function_search_upper_bound=data['objectiveFunctionSearchUpperBound']*2*np.pi,
        objective_function_search_discretization=data['objectiveFunctionSearchDiscretization'],
        plot_lower_bound=data['plotLowerBound']*2*np.pi,
        plot_upper_bound=data['plotUpperBound']*2*np.pi,
        plot_discretization=data['plotDiscretization'],
        genetic_algorithm=genetic_algorithm
    )
    
    return input_data