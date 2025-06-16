##### IMPORTS #####
# Third party imports
import matplotlib.pyplot as plt
# Built-in imports

# Local imports
from .config import *


##### FUNCTIONS DEFINITION #####
def create_simple_graph(x_values, x_title, y_values, y_title, plot_type, graph_title, image_name, image_path):
    plt.figure()
    match plot_type:
        case "scatter":
            plt.scatter(x_values, y_values)
        case _:
            plt.plot(x_values, y_values)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()


def create_multi_y_graph(x_values, x_title, y_values_dict, plot_type, graph_title, image_name, image_path):
    plt.figure()
    for y_title, y_values in y_values_dict.items():
        match plot_type:
            case "scatter":
                plt.scatter(x_values, y_values, label = y_title)
            case "plot":
                plt.plot(x_values, y_values, label = y_title)
            case _:
                print("-W-: For 'create_multi_y_graph' attribute 'plot_type' use: scatter or plot")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()


def create_vertical_graphs(x_values, x_title, y_values_dict, graph_title, image_name, image_path, color):
    plt.figure()
    count = 1
    for y_title, y_values in y_values_dict.items():
        plt.subplot(2, 1, count)
        plt.plot(x_values, y_values, label = y_title, color = color)
        count =+ 1
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()
