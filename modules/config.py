##### IMPORTS #####
# Third party imports
import numpy as np
# Built-in imports
import os
# Local imports


##### GLOBAL VARIABLES #####
# General
run_area = os.getcwd() # Current working directory
#t = np.linspace(0, 20)
# Variables for a mass-spring-damper model
k_range = [1000, 50000] # N/m -> Range for vehicle suspension
b_range = [100, 5000]   # Ns/m -> Range for vehicle suspension
kb_range_dict = {"k": k_range, "b": b_range}
car_mass = 200 # in kg
m = car_mass/4
x_0 = 0
v_0 = 5
#u = signal.unit_impulse(8) # perturbacion (impulso unitario)
u = 0


##### Hyperparameters #####
# General
popu_size = 100  # Population size
generations = 50  # Number of iterations
# Mating
alpha = 0.5
mate_chance = 0.75
parent_popu_size = 25
child_popu_size = 50
# Mutation
mutate_chance = 0.2 
mu = 0.0
sigma = 200
#sigma_k = (k_range[1] - k_range[0]) * 0.05 # 5% of the k range
#sigma_b = (b_range[1] - b_range[0]) * 0.05 # 5% of the b range
