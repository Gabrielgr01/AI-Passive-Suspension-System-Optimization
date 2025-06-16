from scipy.integrate import odeint
from scipy import signal
import numpy as np

from .utils import *
from .config import *

u = 0

def solve_model(t, input_vars):

    k = input_vars[0]
    b = input_vars[1]

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

    return x_sol, v_sol, a_sol

def get_model_maxs(x_sol, v_sol, a_sol):
    sol_maxs_list = []
    sol_list = [x_sol, v_sol, a_sol]
    for sol in sol_list:
        sol_max = np.max(abs(sol))
        sol_maxs_list.append(sol_max)
    return sol_maxs_list

def solve_model_test(t):
    # Grafica de x, v, a vs t
    x, v, a = solve_model( t, [64, 32])
    test_dict = {"x: displacement":x,
                 "v: velocity":v,
                 "a: acceleration":a,}
    create_simple_graph(t, "Time", test_dict, "Model test", "model", run_area)
    #plt.plot(t, x, label = "x", color = "blue")
    #plt.plot(t, v, label = "v", color = "orange")
    #plt.plot(t, a, label = "a", color = "red")
    #plt.legend(loc="upper left")
    #plt.show()

def graph_model_maxs(t):
    k_values = [1, 20, 60, 130]
    b_values = [2, 30, 90, 150]
    k_comb = []
    b_comb = []
    x_points = []
    a_points = []
    for k in k_values:
        for b in b_values:
            x, _, a = solve_model(t, [k, b])
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
