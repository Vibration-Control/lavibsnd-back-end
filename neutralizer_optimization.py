from services.deserialize_optimization_file_service import deserialize_optimization_file
from services.optimization_preparation_service import ga_preparation, optimal_solution
import numpy as np

def neutralizer_optimization(data):
    optimization_input = deserialize_optimization_file(data)
    ga_instance,gene_name,gene_per_neutralizer = ga_preparation(optimization_input)
    
    # Run the GA
    ga_instance.run()

    # Print the best solution found
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Best solution: {solution}, Fitness: {solution_fitness}")
    
    optimal_receptance = optimal_solution(optimization_input, solution, gene_name, gene_per_neutralizer)
    output = 20*np.log10(abs(optimal_receptance))

    return output.tolist()
