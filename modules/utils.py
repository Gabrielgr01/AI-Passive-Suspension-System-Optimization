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
    image_path = str(run_area) + image_dir_name
    if os.path.exists(image_path):
        old_path = str(run_area) + image_dir_name + "_old"
        if os.path.exists(old_path):
            shutil.rmtree(old_path)
        os.replace(image_path, old_path)
        #shutil.rmtree(image_path)
    os.makedirs(image_path)


def create_simple_graph(x_values, x_title, y_values, y_title, annotate_values, plot_type, show_plot, graph_title, image_name, image_path):
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
