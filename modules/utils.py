import matplotlib.pyplot as plt

from .config import *

def create_simple_graph(x_values, x_title, y_values_dict, graph_title, image_name, image_path):
    plt.figure()
    for y_title, y_values in y_values_dict.items():
        plt.plot(x_values, y_values, label = y_title)
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()

def create_vertical_graphs_old(x_values, x_title, y_values_dict, graph_title, image_name, image_path):
    num_subplots = len(y_values_dict)
    fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(8, 4 * num_subplots))
    fig.suptitle(graph_title)

    if num_subplots == 1:
        axes = [axes]  # Validation for when there's only one output

    for ax, (y_title, y_values) in zip(axes, y_values_dict.items()):
        ax.plot(x_values, y_values, label=y_title)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        ax.legend(loc="upper right")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()

def create_vertical_graphs(x_values, x_title, y_values_dict, graph_title, image_name, image_path, color):
    plt.figure()
    count = 1
    for y_title, y_values in y_values_dict.items():
        plt.subplot(2, 1, count)
        plt.plot(x_values, y_values, label = y_title, color = color)
        count =+ count + 1
    #plt.tight_layout()
    plt.legend(loc="upper right")
    plt.xlabel(x_title)
    plt.title(graph_title)
    plt.savefig(f"{image_path}/{image_name}")
    plt.close()
