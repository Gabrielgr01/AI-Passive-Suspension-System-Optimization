##### IMPORTS #####
# Third party imports
from scipy.integrate import odeint
from scipy.signal import find_peaks
# Built-in imports

# Local imports
from .utils import *
from .config import *


##### FUNCTIONS DEFINITION #####
def model(S, t, k, b, u):
    x, v = S
    return [v, (-b*v-k*x+u)/m]


def solve_model(input_vars, t_max, t_samples, u):
    k = input_vars[0]
    b = input_vars[1]
    S_0 = (0, 0)
    t = np.linspace(0, t_max, t_samples)
    solution = odeint(model, y0=S_0, t=t, args=(k, b, u))

    x_sol = solution.T[0]
    v_sol = solution.T[1]
    a_sol = (-b * v_sol - k * x_sol + u) / m

    return t, x_sol, v_sol, a_sol


def get_model_maxs_old(x_sol, v_sol, a_sol):
    sol_maxs_list = []
    sol_list = [x_sol, v_sol, a_sol]
    for sol in sol_list:
        sol_max = np.max(abs(sol))
        sol_maxs_list.append(sol_max)
    return sol_maxs_list


def get_model_maxs(x_sol, v_sol, a_sol, t):
    sol_maxs_list = []

    # Máximo absoluto de desplazamiento y velocidad
    for sol in [x_sol, v_sol]:
        sol_max = np.max(np.abs(sol))
        sol_maxs_list.append(sol_max)

    # Detectar máximos y mínimos locales de la aceleración
    peaks_idx, _ = find_peaks(a_sol)
    valleys_idx, _ = find_peaks(-a_sol)  # Mismos que mínimos locales
    peaks = a_sol[peaks_idx]
    valleys = a_sol[valleys_idx]

    if len(peaks): 
        max_peak = np.max(peaks)
    else:
        max_peak = 0

    if len(valleys):
        min_valley = np.min(valleys)
    else:
        min_valley = 0

    if max_peak > abs(min_valley):
        max_peak_valley = max_peak
        idx = np.where(a_sol==max_peak)
        #print("max_peak: ", max_peak)
    else:
        max_peak_valley = abs(min_valley)
        #max_peak_valley = min_valley
        idx = np.where(a_sol==min_valley)
        #print("min_valley: ", min_valley)
    
    t_a_max = t[idx]
    sol_maxs_list.append(max_peak_valley)
    sol_maxs_list.append(t_a_max)

    return sol_maxs_list


def solve_model_test():
    print("--> Solve Model Test ...")
    # Grafica de x, v, a vs t
    t, x, v, a = solve_model([64, 32], 100, 500, 20000)
    #t, x, v, a = solve_model([1000, 5000], 20, 50, 20000)
    _, _, a_max, t_a_max = get_model_maxs(x, v, a, t)

    ## Mostrar picos y valles de la aceleración
    peaks, _ = find_peaks(a)
    valleys, _ = find_peaks(-a)

    plt.plot(t, a, label="a(t)")
    #plt.plot(t[peaks], a[peaks], "x", label="Picos")
    #plt.plot(t[valleys], a[valleys], "o", label="Valles")
    plt.plot(t_a_max, a_max, "o", label="Max Seleccionado")
    plt.legend()
    plt.xlabel("Tiempo")
    plt.ylabel("Aceleración")
    plt.title("Extremos de la aceleración")
    plt.grid(True)
    plt.show()
    ##

    test_dict = {"x: displacement":x,
                 "v: velocity":v,
                 "a: acceleration":a,}
    create_multi_y_graph(t, "Time", test_dict, "plot", "Model test", "model_test", run_area)


def graph_model_maxs(x_0, v_0, t_max, t_samples):
    #!!! Revisar si es necesaria, sino borrar !!!
    k_values = [1, 20, 60, 130]
    b_values = [2, 30, 90, 150]
    k_comb = []
    b_comb = []
    x_points = []
    a_points = []
    for k in k_values:
        for b in b_values:
            t, x, _, a = solve_model([k, b], t_max, t_samples, u)
            x_max, _, a_max, t_a_max = get_model_maxs(x, _, a, t)
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
