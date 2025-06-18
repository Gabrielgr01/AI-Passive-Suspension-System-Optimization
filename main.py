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

toolbox = base.Toolbox()


def evaluation_function(individual, verbose=False):
    """
    Function:
        Used to evaluate every individual.

        Finds the systems response for a given "k" and "b" values.
        Afterwards, it outputs the maximum magnitude of each
        response (displacement and acceleration).

    Parameters:
        individual (list): list containing the chromosomes "k" and "b".
        verbose (bool): Used to display information about each individuals
                        fitness in the terminal.

    Returns:
        fitness (list): Fitness of a given individual.

    """
    k = individual[0]
    b = individual[1]
    t, x_sol, _, a_sol = model.solve_model([k, b], t_max, t_samples, u)
    x_sol_max, _, a_sol_max, t_a_max = model.get_model_maxs(x_sol, _, a_sol, t)

    if verbose == True:
        print(
            f"Evaluando k={k:.2f}, b={b:.2f}, u={u:.2f} -> x_max={x_sol_max:.2f}, a_max={a_sol_max:.2f}"
        )

    return [x_sol_max, a_sol_max]


def check_bounds(lower_bounds, upper_bounds):
    """
    Function:
        Decorator that checks if the mutated individuals are within
        the allele space. If they are not, the specific chromosome is
        modified to enter said space.

    Parameters:
        lower_bounds (list): list containing the minimum values of "b" and "k".
        upper_bounds (list): list containing the maximum values of "b" and "k".

    Returns:
        function: A decorator that wraps functions in order to maintain
                individuals in the defined allele space.

    Source:
        Found in the official DEAP documentation webpage
        https://deap.readthedocs.io/en/master/tutorials/basic/part2.html

    """

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] < lower_bounds[i]:
                        child[i] = lower_bounds[i]
                    elif child[i] > upper_bounds[i]:
                        child[i] = upper_bounds[i]
            return offspring

        return wrapper

    return decorator


def plot_pareto_front(
    solutions, img_path, annotate_inputs=False, show=False, verbose=False
):
    """
    Function:
        Generates a plot of the Pareto front from a list of non-dominated
        solutions.

    Parameters:
        solutions (list): A list of non-dominated individuals. Each individual
                            is a list [k, b].
        img_path (str): Path where the output image and directory will be
                        created.
        annotate_inputs (bool, optional): If True, annotates each point in the
                                          scatter plot with its (k, b)
                                          values. Default is False.
        show (bool, optional): If True, displays the plot in a window after
                               saving. Default is False.
        verbose (bool, optional): If True, prints details about each solution
                                  evaluated. Default is False.

    Returns:
        None
    """
    
    print("\n--> Getting the Pareto Front ...")

    # Manages directory where images will be created
    image_dir_name = "\\images\\pareto"
    image_path = str(img_path) + image_dir_name
    permission_status = utils.manage_directories_gen(image_dir_name)
    if permission_status == 1:
        return

    # Obtains the Pareto Front
    x_values = []
    a_values = []
    k_values = []
    b_values = []
    annotate_vals = []

    if verbose == True:
        print("- Individuos no dominados:")

    for individual in solutions:
        if verbose == True:
            print(individual)
        x_max, a_max = toolbox.evaluate(individual)
        x_values.append(x_max)  # maximum displacements
        a_values.append(a_max)  # maximum accelerations
        k_values.append(individual[0])  # k values
        b_values.append(individual[1])  # b values

    if annotate_inputs == True:
        annotate_vals = [k_values, b_values]

    utils.create_simple_graph(
        x_values=x_values,
        x_title="Máximo desplazamiento",
        y_values=a_values,
        y_title="Máxima aceleración",
        annotate_values=annotate_vals,
        plot_type="scatter",
        show_plot=show,
        graph_title="Frente de Pareto (Optimización Multiobjetivo)",
        image_name="pareto_front",
        image_path=image_path,
    )
    print("- Saved in: ", image_path)


def get_preferred_solution(solutions, preference, verbose):
    """
    Function:
        Selects the best individual from a list of solutions based on a weighted 
        preference between displacement and acceleration.

    Parameters:
        solutions (list): A list of non-dominated individuals. Each individual
                          is a list [k, b].
        preference (float): A float between 0 and 1 indicating the preference for the displacement. The second objective 
                            The acceleration is weighted as (1 - preference).
        verbose (bool): If True, print detailed information about the selected individual 
                        and the applied preference.

    Returns:
        best_individual (list): The solution with the lowest weighted score.
        best_inputs (list): The chromosome values of selected individual [k, b].
        best_outputs (list): The evaluated objective values [x_max, a_max].
    """
    
    print("\n--> Getting Prefered Solution ...")
    # 'preference' will be applied to the first output of the evaluation_function
    # '(1 - preference)' will be applied to the second output of the evaluation_function

    best_score = float("inf")
    best_individual = None
    best_outputs = []
    best_inputs = []

    for individual in solutions:
        x_max, a_max = toolbox.evaluate(individual)
        score = preference * x_max + (1 - preference) * a_max
        if score < best_score:
            best_outputs = [x_max, a_max]
            best_score = score
            best_individual = individual

    best_inputs.append(best_individual[0])  # k
    best_inputs.append(best_individual[1])  # b

    if verbose == True:
        print(
            "- Using preference: ",
            int(preference * 100),
            r"% for displacement. ",
            int((1 - preference) * 100),
            r"% for acceleration.",
        )
        print("- Best individual:\n\tk = ", best_inputs[0], " b = ", best_inputs[1])
        print(
            "- Output:\n\tDisplacement: ",
            best_outputs[0],
            ". Acceleration: ",
            best_outputs[1],
        )

    return best_individual, best_inputs, best_outputs


def run_evolutionary_algorithm():
    """
    Function:
        Runs a multi-objective evolutionary algorithm using the DEAP library.

        This function sets up and executes an evolutionary process to optimize two 
        conflicting objectives (displacement and acceleration). The function registers 
        genetic operators, initializes the population, executes the evolution, and 
        visualizes the results.

    Parameters:
        None

    Returns:
        None
    """
    
    print("\n--> Running the Evolutionary Algorithm ...")

    # Negative weights for minimization
    pesos_fitness = (
        -1.0,
        -1.0,
    )

    # Fitness function definition
    creator.create("fitness_function", base.Fitness, weights=pesos_fitness)
    # Individual definition
    creator.create("individual", list, fitness=creator.fitness_function, typecode="f")

    # Alleles
    toolbox.register(
        "k_stiffness", random.uniform, a=k_range[0], b=k_range[1]
    )  # Stiffness 'k' gene
    toolbox.register(
        "b_cushioning", random.uniform, a=b_range[0], b=b_range[1]
    )  # Cushioning 'b' gene
    # Individual generator
    toolbox.register(
        "individual_generation",
        tools.initCycle,
        creator.individual,
        (toolbox.k_stiffness, toolbox.b_cushioning),
        n=1,
    )
    # Population generator
    toolbox.register(
        "population", tools.initRepeat, list, toolbox.individual_generation
    )
    toolbox.register("evaluate", evaluation_function)
    # Evolution operators
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", tools.cxBlend, alpha=alpha)
    toolbox.register(
        "mutate", tools.mutGaussian, mu=mu, sigma=(sigma_k, sigma_b), indpb=0.2
    )
    toolbox.decorate(
        "mate", check_bounds([k_range[0], b_range[0]], [k_range[1], b_range[1]])
    )
    toolbox.decorate(
        "mutate", check_bounds([k_range[0], b_range[0]], [k_range[1], b_range[1]])
    )

    if debug == True:
        # Test for the population and individuals generation
        population_test = toolbox.population(n=popu_size)
        individual_test = toolbox.individual_generation()
        print("Individuo: ", individual_test)
        print("Ejemplo de poblacion: ", population_test)

    # Statistics on the general fitness of the population
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)  # Generation 'Average'
    stats.register("std", np.std)  # Individuals 'Standard Deviation'
    stats.register("min", np.min)  # 'Min Fitness' of the generation
    stats.register("max", np.max)  # 'Max Fitness' of the generation

    hof = tools.ParetoFront()  # Hall of Fame
    popu = toolbox.population(n=popu_size)  # Defines the initial population
    
    # Runs the Evolutionary Algorithm
    popu, logbook = algorithms.eaMuPlusLambda(
        population=popu,
        toolbox=toolbox,
        mu=parent_popu_size,
        lambda_=child_popu_size,
        cxpb=mate_chance,
        mutpb=mutate_chance,
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose,
    )
    
    # Saves/shows the pareto front
    plot_pareto_front(
        solutions=hof,
        img_path=run_area,
        annotate_inputs=display_annotations,
        show=display_plots,
    )

    # Gets/prints the preferred solution
    get_preferred_solution(
        solutions=hof, preference=x_to_a_preference, verbose=True
    )
    print("")


##### MAIN EXECUTION #####

if debug == True:
    # Test for the Evaluation Function.
    # Individual with greater k and b values should have smaller 
    # x_max and a_max values.
    print("Evaluación:", evaluation_function([30000, 1000]))  
    print("Evaluación (peor caso):", evaluation_function([1000, 100]))

# Problem Analysis
model.test_model(
    input_vars=[64, 32],
    t_max=t_max,
    t_sample=t_samples,
    u=u,
    img_path=run_area,
    show_plots=display_plots,
)
analysis.ceteris_paribus(
    n_points=10, inputs_range_dict=kb_range_dict, scale_porcentage=0.2
)

# Evolutionary Algorithm
run_evolutionary_algorithm()
