##### IMPORTS #####
# Third party imports
from scipy.integrate import odeint
from scipy import signal
import numpy as np
# Built-in imports

# Local imports
from .utils import *
from .config import *


##### FUNCTIONS DEFINITION #####
def solve_model(input_vars, t_max, t_samples):
    k = input_vars[0]
    b = input_vars[1]
    t = np.linspace(0, t_max, t_samples)

    # System transfer function for displacement
    num = [1/m]
    den = [1, b/m, k/m]
    system = signal.TransferFunction(num, den)

    # Find step response of displacement
    t, x_sol = signal.step(system, T=t)

    # System transfer function for velocity
    num = [1/m, 0]
    den = [1, b/m, k/m]
    system = signal.TransferFunction(num, den)

    # Find step response for velocity
    t, v_sol = signal.step(system, T=t)

    # System transfer function for acceleration
    num = [1/m, 0, 0]
    den = [1, b/m, k/m]
    system = signal.TransferFunction(num, den)

    # Find step response for acceleration
    t, a_sol = signal.step(system, T=t)

    return t, x_sol, v_sol, a_sol


def get_model_maxs(x_sol, v_sol, a_sol):
    sol_maxs_list = []
    x_max = np.max(abs(x_sol))
    v_max = np.max(abs(v_sol))
    a_max = np.min(a_sol)
    sol_maxs_list.append(x_max)
    sol_maxs_list.append(v_max)
    sol_maxs_list.append(a_max)
    return sol_maxs_list


def solve_model_test():
    # Grafica de x, v, a vs t
    t, x, v, a = solve_model([64, 32], 20, 50)
    test_dict = {"x: displacement":x,
                 "v: velocity":v,
                 "a: acceleration":a,}
    create_multi_y_graph(t, "Time", test_dict, "plot", "Model test", "model_test", run_area)


def graph_model_maxs(t_max, t_samples):
    #!!! Revisar si es necesaria, sino borrar !!!
    k_values = [1, 20, 60, 130]
    b_values = [2, 30, 90, 150]
    k_comb = []
    b_comb = []
    x_points = []
    a_points = []
    for k in k_values:
        for b in b_values:
            t, x, _, a = solve_model([k, b], t_max, t_samples)
            x_max, _, a_max = get_model_maxs(x, _, a)
            k_comb.append(k)
            b_comb.append(b)
            x_points.append(x_max)
            a_points.append(a_max)
    #print(x_points)
    #print(a_points)
    #print("x size: "+str(len(x_points)))

    t_maxs = np.linspace(0, len(x_points), len(x_points))
    #print("t size: "+str(len(t_maxs)))
    plt.plot(t_maxs, x_points, label = "x", color = "green")
    plt.plot(t_maxs, a_points, label = "a", color = "purple")
    for point in range(len(t_maxs)):
        t = t_maxs[point]
        x = x_points[point]
        a = a_points[point]
        k = k_comb[point]
        b = b_comb[point]
        #plt.annotate(f"k:{k},b:{b}", xy=(t, x))
        plt.annotate(f"k:{k},b:{b}", xy=(t, a))
    plt.legend(loc="upper left")
    plt.show()
