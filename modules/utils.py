##### IMPORTS #####

# Third party imports
import matplotlib.pyplot as plt

# Built-in imports
import stat
import shutil

# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####

def print_files_tree():
    """
    Function: Prints the hierarchical tree structure of the main program files and modules.
    Used to explain the code in the written report.
    
    Parameters:
        None

    Returns:
        None
    """
    
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


def manage_directories_gen_old(image_dir_name):
    """
    Function: Manages image directories by moving the existing directory to a backup (_old)
    and creating a new directory with the original name.

    Parameters:
        image_dir_name (str): Name of the image to generate.

    Returns:
        int: 0 if successful, 1 if PermissionError occurrs.
    """
    
    image_path = str(run_area) + image_dir_name
    try:
        if os.path.isdir(image_path):
            old_path = str(run_area) + image_dir_name + "_old"
            if os.path.isdir(old_path):
                os.remove(old_path)
            os.replace(image_path, old_path)
            os.chmod(old_path, stat.S_IWRITE)
            os.makedirs(image_path)
            os.chmod(image_path, stat.S_IWRITE)
        else:
            os.makedirs(image_path)
            os.chmod(image_path, stat.S_IWRITE)
        return 0
    except PermissionError:
        print("-E-: 'Access Denied' while trying to modify directories. Files not created.")
        print(f"-I-: Possible affected directories:\n\t{image_path}\n\t{old_path}")
        print("-I-: Solution: Execute program as administrator or remove the affected directories.\n")
        return 1


def manage_directories_gen(image_dir_name):
    """
    Function: Manages image directories by moving the existing directory to a backup (_old)
    and creating a new directory with the original name.

    Parameters:
        image_dir_name (str): Name of the image to generate.

    Returns:
        None
    """
    
    image_path = str(run_area) + image_dir_name
    if os.path.exists(image_path):
        old_path = str(run_area) + image_dir_name + "_old"
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
        os.replace(image_path, old_path)
        #shutil.rmtree(image_path)
    os.makedirs(image_path)


def create_simple_graph(x_values, x_title, y_values, y_title, annotate_values, plot_type, show_plot, graph_title, image_name, image_path):
    """
    Function: Creates and saves a 2D graph (scatter or line).

    Parameters:
        x_values (list): Data points for the x-axis.
        x_title (str): Label for the x-axis.
        y_values (list): Data points for the y-axis.
        y_title (str): Label for the y-axis.
        annotate_values (list): Optional. List of two lists [k_values, b_values] for annotating each (k, b) point.
        plot_type (str): Type of plot to create. Valid options: 'scatter', others default to line plot.
        show_plot (bool): Shows the plot while running if True.
        graph_title (str): Title of the graph.
        image_name (str): Name of the image to generate.
        image_path (str): Path to save the image.

    Returns:
        None
    """
    
    plt.figure()
    match plot_type:
        case "scatter":
            plt.scatter(x_values, y_values)
        case _:
            plt.plot(x_values, y_values)

    # Annotate each point with k and b values
    if len(annotate_values) == 2:
        k_values = annotate_values[0]
        b_values = annotate_values[1]
        for i in range(len(x_values)):
            plt.annotate(f"k={k_values[i]:.1f}\nb={b_values[i]:.1f}",
                         (x_values[i], y_values[i]),
                         textcoords="offset points",
                         xytext=(5,5),
                         ha='left',
                         fontsize=8)
    
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    if show_plot == True:
        plt.show()
    plt.close()


def create_multi_y_graph(x_values, x_title, y_values_dict, plot_type, show_plot, graph_title, image_name, image_path):
    """
    Function: Creates and saves a 2D graph with multiple y-axis functions (scatter or line).

    Parameters:
        x_values (list or array): Data points for the x-axis.
        x_title (str): Label for the x-axis.
        y_values_dict (dict): Dictionary where keys are labels and values are lists of y data.
        plot_type (str): Type of plot to create. Valid options: 'scatter', others default to line plot.
        show_plot (bool): Shows the plot while running if True.
        graph_title (str): Title of the graph.
        image_name (str): Name of the image to generate.
        image_path (str): Path to save the image.

    Returns:
        None
    """
    
    plt.figure()
    for y_title, y_values in y_values_dict.items():
        match plot_type:
            case "scatter":
                plt.scatter(x_values, y_values, label = y_title)
            case "plot":
                plt.plot(x_values, y_values, label = y_title)
            case _:
                print("-W-: For 'create_multi_y_graph' attribute 'plot_type' use: scatter or plot")
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    if show_plot == True:
        plt.show()
    plt.close()


def create_vertical_graphs(x_values, x_title, y_values_dict, show_plot, graph_title, image_name, image_path, color):
    """
    Function: Creates and saves a subplots arranged in vertical way
        Subplots share the same x-axis.

    Parameters:
        x_values (list or array): Data points for the x-axis.
        x_title (str): Label for the x-axis.
        y_values_dict (dict): Dictionary where keys are labels and values are lists of y data.
        show_plot (bool): Shows the plot while running if True.
        graph_title (str): Title of the graph.
        image_name (str): Name of the image to generate.
        image_path (str): Path to save the image.
        color (str): Color for the plot lines.

    Returns:
        None
    """
    
    plt.figure()
    count = 1
    for y_title, y_values in y_values_dict.items():
        plt.subplot(2, 1, count)
        plt.plot(x_values, y_values, label = y_title, color = color)
        count += 1
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    if show_plot == True:
        plt.show()
    plt.close()
