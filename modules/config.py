##### IMPORTS #####

# Third party imports

# Built-in imports
import os

# Local imports


##### GLOBAL VARIABLES #####

# General
run_area = os.getcwd()          # Current working directory.
verbose = False                 # To print detailed messages in terminal.
display_plots = False           # To display plots while running algorithm.
display_annotations = False     # To display annotations in plots.
debug = False                   # To print messages useful for debbuging.

# Mass-spring-damper model
k_range = [1000, 50000]         # Stiffness (k) [N/m] -> Range for vehicle suspension.
b_range = [100, 5000]           # Cushioning (b) [Ns/m] -> Range for vehicle suspension.
kb_range_dict = {"k": k_range,
                 "b": b_range}  # Dict with k and b ranges.
car_mass = 1000                 # Car mass [kg].
m = car_mass/4                  # Mass for each suspension.
u = 5000                        # External force (step) - Perturbation.
x_units = "m"                   # Units for the output displacement (x).
v_units = "m/s"                 # Units for the output velocity (v).
a_units = "m/s^2"               # Units for the output acceleration (a).
t_max = 100                     # Maximum time for simulation.
t_samples = 500                 # Number of time samples.


##### HYPERPARAMETERS #####

# General
popu_size = 100         # Population size.
generations = 50        # Number of iterations.
# Mating
alpha = 0.5             # Blend factor.
mate_chance = 0.75      # Chance that a pair of parents will actually mate.
parent_popu_size = 25   # Number of individuals selected to become parents.
child_popu_size = 25    # Number of offspring that are produced during mating.
# Mutation
mutate_chance = 0.2     # Probability that an individual of gene will mutate.
mu = 0.0                #  Mean of the Gaussian distribution. Zero -> symmetric perturbation.
sigma_k = (k_range[1] - k_range[0]) * 0.05  # 5% of the k range.
sigma_b = (b_range[1] - b_range[0]) * 0.05  # 5% of the b range.
# Pareto
x_to_a_preference = 0.7 # Used to select the 'preferred solution'
