##### IMPORTS #####
# Third party imports
import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
# Built-in imports
import random
# Local imports
from modules.config import *
import modules.utils as utils
import modules.model as model
import modules.analysis as analysis


##### FUNCTIONS DEFINITION #####
def print_files_tree():
    print ("""
Program File Tree:
C:.
│   main.py
│
└───modules
        analysis.py
        config.py
        model.py
        utils.py
        __init__.py
       """)

def evaluation_function(individual):
    global x_0, v_0, t
    k = individual[0]
    b = individual[1]
    t, x_sol, _, a_sol = model.solve_model(x_0, v_0, [k, b], 20, 50)
    x_sol_max, _, a_sol_max = model.get_model_maxs(x_sol, _, a_sol)
    return [x_sol_max, a_sol_max]


##### Problem Analysis #####
model.solve_model_test()
#model.graph_model_maxs(x_0, v_0, 20, 50)
analysis.ceteris_paribus(10, kb_range_dict, 0.2)


##### EVOLUTIONARY ALGORITHM #####
print("--> Running the Evolutionary Algorithm ...")
pesos_fitness = (-1., -1.,)  # Negative weights for minimization
toolbox = base.Toolbox()

# 
creator.create("fitness_function", base.Fitness, weights=pesos_fitness)
creator.create("individual",
               list,
               fitness=creator.fitness_function,
               typecode="f")

# Alleles
toolbox.register("k_stiffness", random.uniform, a=k_range[0], b=k_range[1])  # Stiffness 'k' gene
toolbox.register("b_cushioning", random.uniform, a=b_range[0], b=b_range[1]) # Cushioning 'b' gene
# 
toolbox.register("individual_generation",
                 tools.initCycle,
                 creator.individual,
                 (toolbox.k_stiffness, toolbox.b_cushioning),
                 n=1)
toolbox.register("population",
                 tools.initRepeat,
                 list,
                 toolbox.individual_generation)
toolbox.register("evaluate", evaluation_function)
# Evolution operators
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", tools.cxBlend, alpha=alpha)
toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=0.2)

## Prueba del generador de individuos
#ejemploPopu = toolbox.population(n=popu_size)
#ejemplo = toolbox.individual_generation()
#print("Individuo: ", ejemplo)
#print("Ejemplo de poblacion: ", ejemploPopu)

# Statistics on the general fitness of the population
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)  # Generation 'Average'
stats.register("std", np.std)   # Individuals 'Standard Deviation'
stats.register("min", np.min)   # 'Min Fitness' of the generation
stats.register("max", np.max)   # 'Max Fitness' of the generation

hof = tools.ParetoFront() # Hall of Fame
popu = toolbox.population(n=popu_size) # Defines the initial population
popu, logbook = algorithms.eaMuPlusLambda(
    population=popu, toolbox=toolbox, mu=parent_popu_size,
    lambda_=child_popu_size, cxpb=mate_chance, mutpb=mutate_chance,
    ngen=generations, stats=stats, halloffame=hof
    ) # Runs the Evolutionary Algorithm

# Obtains the Pareto Front

#print('------------------------')
#print("Individuos no dominados:")
#for item in hof:
#    print(item)
#    results = toolbox.evaluate(item)
#    plt.scatter(results[0], results[1])
#    plt.xlabel('Parametro X')
#    plt.ylabel('Parametro Y')
#    plt.title('Frente de Pareto')
#    plt.show()

x_vals = []
a_vals = []
k_vals = []
b_vals = []

print('------------------------')
print("Individuos no dominados:")
for item in hof:
    print(item)
    results = toolbox.evaluate(item)
    x_vals.append(results[0])  # desplazamiento máximo
    a_vals.append(results[1])  # aceleración máxima
    k_vals.append(item[0])
    b_vals.append(item[1])

plt.figure(figsize=(10, 6))
plt.scatter(x_vals, a_vals, color='blue', label='Frente de Pareto')

# Escala logarítmica opcional
#plt.xscale("log")
#plt.yscale("log")

# Anotar cada punto con k y b
for i in range(len(x_vals)):
    plt.annotate(f"k={k_vals[i]:.1f}\nb={b_vals[i]:.1f}",
                 (x_vals[i], a_vals[i]),
                 textcoords="offset points",
                 xytext=(5,5),
                 ha='left',
                 fontsize=8)

plt.xlabel("Máximo desplazamiento")
plt.ylabel("Máxima aceleración")
plt.title("Frente de Pareto (Optimización multiobjetivo)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
