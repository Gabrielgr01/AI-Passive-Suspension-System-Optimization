# Librerías importadas
import random
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from deap import creator, base, tools, algorithms

# Hiperparámetros
popu_size = 100  # Tamaño de población
generations = 35  # Número de iteraciones

# Hiperparámetros de cruze
alpha = 40
mate_chance = 0.5
parent_popu_size = 50
child_popu_size = 100

# Hiperparámetros de mutación
mutate_chance = 0.3
mu = 0.0
sigma = 1

# Cinematica y variables del sistema
car_mass = 200 # in kg
m = car_mass/4
x_0 = 0
v_0 = 5
#u = signal.unit_impulse(8) # perturbacion (impulso unitario)
u = 0
t = np.linspace(0, 20)


# Pesos positivos implican maximización, negativos minimización
pesos_fitness = (1., 1.,)  # Pesos de los parámetros a optimizar

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
toolbox.register("k_stiffness", random.uniform, a=0, b=200)
toolbox.register("b_cushioning", random.uniform, a=0, b=200)

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

# Prueba del generador de individuos
ejemploPopu = toolbox.population()
ejemplo = toolbox.individual_generation()
print("Individuo: ", ejemplo)
print("Ejemplo de poblacion: ", ejemploPopu)


def model(t, S, k, b):
    x, v = S
    return [v, (-b*v-k*x+u)/m]


def solve_model(x_0, v_0, t, k, b):
    S_0 = (x_0, v_0)
    solution = odeint(model, y0=S_0, t=t, tfirst=True, args=(k, b))

    x_sol = solution.T[0]
    v_sol = solution.T[1]
    a_sol = (-b * v_sol - k * x_sol + u) / m

    return x_sol, v_sol, a_sol


def solve_model_test(t):
    # Grafica de x, v, a vs t
    x, v, a = solve_model(0, 5, t, 64, 32)
    plt.plot(t, x, label = "x", color = "blue")
    plt.plot(t, v, label = "v", color = "orange")
    plt.plot(t, a, label = "a", color = "red")
    plt.legend(loc="upper left")
    plt.show()

solve_model_test(t)


def evaluation_function(individual):
    # !!!!!! Esta dando problemas porque retorna vectores. 
    # La funcion de evaluacion debe de retornar escalares o floats
    global x_0, v_0, t
    k = individual[0]
    b = individual[1]
    x_sol, _, a_sol = solve_model(x_0, v_0, t, k, b)
    return [x_sol, a_sol]

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
print('------------------------')
print("Individuos no dominados:")
for item in hof:
    print(item)
    results = toolbox.evaluate(item)
    plt.scatter(results[0], results[1])
    plt.xlabel('Parametro X')
    plt.ylabel('Parametro Y')
    plt.title('Frente de Pareto')
    plt.show()

