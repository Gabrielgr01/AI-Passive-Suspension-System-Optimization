##### IMPORTS #####

# Third party imports
import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks

# Built-in imports

# Local imports
from .utils import *
from .config import *


##### FUNCTIONS DEFINITION #####

def model(S, t, k, b, u):
    """
    Function: 
        Define the differential equations for a mass-spring-damper system.

    Parameters:
        S (list): State vector / Initial conditions [position x, velocity v].
        t (float): Time variable (not used explicitly but required by odeint).
        k (float): Spring constant.
        b (float): Damping coefficient.
        u (float): External force applied.

    Returns:
        List: Derivatives [dx/dt, dv/dt].
    """
    
    x, v = S
    return [v, (-b * v - k * x + u) / m]


def solve_model(input_vars, t_max, t_samples, u):
    """
    Function: 
        Solve the mass-spring-damper system using the given parameters.

    Parameters:
        input_vars (list): List with constants: k (stiffness) and b (cushioning).
        t_max (float): Maximum time for simulation.
        t_samples (int): Number of time samples.
        u (float): External force applied.

    Returns:
        List: time (t), displacement (x), velocity (v), and acceleration a.
    """
    
    k = input_vars[0]
    b = input_vars[1]
    S_0 = (0, 0)
    t = np.linspace(0, t_max, t_samples)
    solution = odeint(model, y0=S_0, t=t, args=(k, b, u))

    x_sol = solution.T[0]
    v_sol = solution.T[1]
    a_sol = (-b * v_sol - k * x_sol + u) / m

    return t, x_sol, v_sol, a_sol


def get_model_maxs(x_sol, v_sol, a_sol, t):
    """
    Function: 
        Obtain maximum absolute values for displacement, velocity, and acceleration.

    Parameters:
        x_sol (list): Max displacement over time.
        v_sol (list): Max velocity over time.
        a_sol (list): max acceleration over time.
        t (float): Time at max acceleration.

    Returns:
        List: [max abs displacement, max abs velocity, max abs acceleration, time of max acceleration].
    """

    sol_maxs_list = []

    # Abs max for displacement and velocity
    for sol in [x_sol, v_sol]:
        sol_max = np.max(np.abs(sol))
        sol_maxs_list.append(sol_max)

    # Detect local max y min for the acceleration
    peaks_idx, _ = find_peaks(a_sol) # Local max
    valleys_idx, _ = find_peaks(-a_sol)  # Local min
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
        idx = np.where(a_sol == max_peak)
    else:
        max_peak_valley = abs(min_valley)
        # max_peak_valley = min_valley
        idx = np.where(a_sol == min_valley)

    t_a_max = t[idx]
    sol_maxs_list.append(max_peak_valley)
    sol_maxs_list.append(t_a_max)

    return sol_maxs_list


def test_model(input_vars, t_max, t_sample, u, img_path, show_plots):
    """
    Function: 
        Test the mass-spring-damper model and generate plots for:
        x(t), v(t), a(t),
        peaks/valleys for the acceleration and
        max absolute acceleration.

    Parameters:
        input_vars (list): input_vars (list): List with constants: k (stiffness) and b (cushioning).
        t_max (float): Maximum time for simulation.
        t_sample (int): Number of time samples.
        u (float): External force applied.
        img_path (str): Path to save the plots.
        show_plots (bool): Whether to display plots while running.

    Returns:
        None
    """

    print("--> Testing model ...")

    # Manages directory where images will be created
    image_dir_name = "\\images\\model_tests"
    image_path = img_path + image_dir_name
    permission_status = manage_directories_gen(image_dir_name)
    if permission_status == 1:
        return

    # Plot of x, v, a vs t
    t, x, v, a = solve_model(input_vars, t_max, t_sample, u)
    _, _, a_max, t_a_max = get_model_maxs(x, v, a, t)

    test_dict = {
        "x: Desplazamiento " + x_units: x,
        "v: Velocidad " + v_units: v,
        "a: Aceleración " + a_units: a,
    }
    create_multi_y_graph(
        t,
        "Tiempo (s)",
        test_dict,
        "plot",
        show_plots,
        "Modelo Masa-Resorte-Amortiguador",
        "model_test",
        image_path,
    )

    # To show peaks and valleys of the acceleration
    peaks, _ = find_peaks(a)
    valleys, _ = find_peaks(-a)

    plt.plot(t, a, label="a(t)")
    plt.plot(t[peaks], a[peaks], "x", label="Picos")
    plt.plot(t[valleys], a[valleys], "o", label="Valles")
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Aceleración " + a_units)
    plt.title("Picos y valles de la aceleración")
    plt.grid(True)
    plt.savefig(f"{image_path}/model_peaks_test")
    if show_plots == True:
        plt.show()
    plt.close()

    # To show max abs peak for the acceleration
    plt.plot(t, a, label="a(t)")
    plt.plot(t_a_max, a_max, "o", label="Max Abs Seleccionado")
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Aceleración " + a_units)
    plt.title("Maximo absoluto de la aceleración")
    plt.grid(True)
    plt.savefig(f"{image_path}/model_max_test")
    if show_plots == True:
        plt.show()
    plt.close()
