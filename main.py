# Librerías importadas
import os
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from deap import creator, base, tools, algorithms

#import modules.config as config
from modules.config import *
import modules.utils as utils
import modules.model as model
import modules.analysis as analysis


def evaluation_function(individual):
    global x_0, v_0, t
    k = individual[0]
    b = individual[1]
    x_sol, _, a_sol = model.solve_model(t, [k, b])
    x_sol_max, _, a_sol_max = model.get_model_maxs(x_sol, _, a_sol)
    return [x_sol_max, a_sol_max]


#model.solve_model_test(t)
#graph_model_maxs(x_0, v_0, t)
#analysis.ceteris_paribus(10, kb_range_dict, 0.2)



# Pesos positivos implican maximización, negativos minimización
pesos_fitness = (-1., -1.,)  # Pesos de los parámetros a optimizar

# Definición del toolbox
toolbox = base.Toolbox()

# Definición de la función fitness que manejará el toolbox
creator.create("fitness_function", base.Fitness, weights=pesos_fitness)

# Definición de la clase a contener los  individuos
creator.create("individual",
               list,
               fitness=creator.fitness_function,
               typecode="f")

# Definición de los alelos
toolbox.register("k_stiffness", random.uniform, a=k_range[0], b=k_range[1])
toolbox.register("b_cushioning", random.uniform, a=b_range[0], b=b_range[1])


# Definicion de la generacion de individuos
toolbox.register("individual_generation",
                 tools.initCycle,
                 creator.individual,
                 (toolbox.k_stiffness, toolbox.b_cushioning),
                 n=1)

# Definición del método de generación de la población
toolbox.register("population",
                 tools.initRepeat,
                 list,
                 toolbox.individual_generation,
                 n = popu_size)

## Prueba del generador de individuos
#ejemploPopu = toolbox.population()
#ejemplo = toolbox.individual_generation()
#print("Individuo: ", ejemplo)
#print("Ejemplo de poblacion: ", ejemploPopu)


# Registro de la función de evaluación
# NOTE: al ser multiobjetivo, hay que alimentarle una salida por peso.
toolbox.register("evaluate", evaluation_function)
# Definición del método de cruce.
toolbox.register("mate", tools.cxBlend, alpha=alpha)
# Definición del método de mutación
toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=0.2)
# Definición del método de selección.
toolbox.register("select", tools.selNSGA2)

# Estadísticas del fitness general de la población
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean) # Promedio de la gen
stats.register("std", np.std) # Desviación estándar de los individuos
stats.register("min", np.min) # Fitness mínimo de la gen
stats.register("max", np.max) # Fitness máximo de la gen
# Hall of Fame: presentación del Frente de Pareto
hof = tools.ParetoFront()
# Una vez que todo está registrado y establecido, ya se puede comenzar
# a generar una población y evaluarla.
popu = toolbox.population(n=popu_size)
# La función a continuación corre el Algoritmo Evolutivo según sus parámetros
popu, logbook = algorithms.eaMuPlusLambda(
    population=popu, toolbox=toolbox, mu=parent_popu_size,
    lambda_=child_popu_size, cxpb=mate_chance, mutpb=mutate_chance,
    ngen=generations, stats=stats, halloffame=hof
    )

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
